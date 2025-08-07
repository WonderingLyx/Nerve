import os 
import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import zoom  # 用于 3D 图像的重采样
import tifffile as tiff
from PIL import Image

#from myutils import *

# local test
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dataset import FullDataset

config_path = '/mnt/40B2A1DBB2A1D5A6/lyx/project/MedSam2/SAM-Unet/CONFIG/SAM2-Unet_GPS.yaml'
config = OmegaConf.load(config_path)

CUBE_SIZE = 150

def iou(pre, label):
    eps = 1e-8
    if pre.sum() == 0:
        raise ValueError('Wrong threshold make whole zeros')
    return (pre * label).sum() / ((pre + label).sum() - (pre * label).sum() + eps)


def t_iou(pre, label):
    eps = 1e-8

    label_ori = label.clone()
    
    label_batch_size = label.clone()
    p1 = torch.nn.functional.max_pool3d(label_batch_size, (3, 1, 1), 1, (1, 0, 0))
    p2 = torch.nn.functional.max_pool3d(label_batch_size, (1, 3, 1), 1, (0, 1, 0))
    p3 = torch.nn.functional.max_pool3d(label_batch_size, (1, 1, 3), 1, (0, 0, 1))
    label = torch.max(torch.max(p1, p2), p3) - label_ori

    return (pre * (label_ori + label)).sum() / ((pre * label + label_ori + pre - label_ori * pre).sum() + eps)

def dice_error(input, target):
    smooth = 1.
    num = input.size(0)
    m1 = input.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def resize_volume(volume, target_shape):
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.ndim} dimensions")
    depth_factor = target_shape[0] / volume.shape[0]
    height_factor = target_shape[1] / volume.shape[1]
    width_factor = target_shape[2] / volume.shape[2]
    resized = zoom(volume, (depth_factor, height_factor, width_factor), order=3)
    return resized

def apply_transformations(reconstructed_image):
    """
    对恢复的3D图像进行顺时针旋转90度、轴对称翻转和逆时针旋转180度。
    
    Parameters:
        reconstructed_image (np.ndarray): 恢复后的3D图像 (depth, height, width)。
    
    Returns:
        np.ndarray: 经过变换后的3D图像。
    """
    # 顺时针旋转90度：沿轴(1, 2)（高度和宽度）旋转
    rotated_image = np.rot90(reconstructed_image, k=-1, axes=(1, 2))

    # 轴对称翻转：沿高度轴对称
    flipped_image = np.flip(rotated_image, axis=1)

    # 逆时针旋转180度：再次沿 (1, 2) 旋转两次
    final_image = np.rot90(flipped_image, k=2, axes=(1, 2))

    return final_image

def load_tiff_stack(file_path):
    """加载一个3D TIFF文件，返回一个NumPy数组"""
    images = []
    with Image.open(file_path) as img:
        for i in range(img.n_frames):
            img.seek(i)
            images.append(np.array(img))
    return np.stack(images, axis=0)  # 堆叠形成3D数组


def compute_3d_mse_loss(axial_volume, coronal_reconstructed_volume):
    diff1 = axial_volume - coronal_reconstructed_volume
    diff2 = coronal_reconstructed_volume - axial_volume
    mse_loss = (torch.mean(diff1 ** 2) + torch.mean(diff2 ** 2)) / 2.0
    return mse_loss



def save_predict_tiff(pred_3d_np, view:str, image_index:int, output_path:str):
    
    if pred_3d_np.ndim == 4: #*batch 处理 (150, batches, 352, 352)
        pred_3d_np = np.concatenate(pred_3d_np)

    if pred_3d_np.ndim > 4:
        raise ValueError('Wrong dim')
    
    assert pred_3d_np.shape[0] == CUBE_SIZE

    pred_3d_resized = resize_volume(pred_3d_np, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
    pred_3d_resized = (pred_3d_resized * 255).astype(np.uint8)
    
    #* 如果是 Coronal 视角，执行重建操作
    if view == 'coronal':
        name=f"Image_{image_index}_{view}"
        pred_filename = os.path.join(output_path, f"{name}.tiff")

        width = CUBE_SIZE
        #* 重构：从堆叠的深度恢复为原始深度
        reconstructed = np.zeros((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), dtype=pred_3d_resized.dtype)
        for i in range(width):
            reconstructed[:, :, i] = pred_3d_resized[i]
        #* 应用旋转、翻转变换
        final_image = apply_transformations(reconstructed)  
        tiff.imwrite(pred_filename, final_image, photometric='minisblack')
    
    if view == 'axial':
        #* 保存pred axial
        name=f"Image_{image_index}_{view}"
        pred_filename = os.path.join(output_path, f"{name}.tiff")
        tiff.imwrite(pred_filename, pred_3d_resized, photometric='minisblack')

    
@torch.no_grad()
def save_predics(data_loader, model, device, epoch, view, output_path, data_type='train', metrics=['acc'], use_num=None):
    
    #*set output dir
    assert os.path.exists(output_path)
    output_path = os.path.join(output_path, 'Epoch_' + str(epoch))
    os.makedirs(output_path, exist_ok=True)

    #* switch to eval mode
    #model.eval()
    
    pred_slices = []
    batch_size = data_loader.batch_size
    img_index = 0
    for idx, batch in enumerate(tqdm(data_loader, desc=f'evaluate process @Epoch {epoch} @view {view}...')):
        image_id = batch['image_index'].unique().item()
        if image_id != img_index:
            pred_3d = torch.stack([torch.sigmoid(pred).detach().cpu() for pred in pred_slices])  # Shape: [depth, height, width]
            pred_3d_np = pred_3d.numpy().squeeze()  # 去掉多余的维度 ([depth, height, width])
            save_predict_tiff(pred_3d_np, view, img_index, output_path)

            #* 转换index
            img_index = image_id
            pred_slices = []

        x = batch['image'].to(device)
        pred, pred_1, pred_2 = model(x)
        pred_slices.append(pred)

    if len(pred_slices) != 0:
        pred_3d = torch.stack([torch.sigmoid(pred).detach().cpu() for pred in pred_slices])  # Shape: [depth, height, width]
        pred_3d_np = pred_3d.numpy().squeeze()  # 去掉多余的维度 ([depth, height, width])
        save_predict_tiff(pred_3d_np, view, img_index, output_path)

    
def evaluate(data_dir:str, epoch:int, label_dataset, device,  preds_type='axial'):

    assert os.path.exists(data_dir)
    data_dir = os.path.join(data_dir, 'Epoch_'+str(epoch))

    files = os.listdir(data_dir)
    axial_volume_list = [f for f in files if f.split('.')[0].endswith('axial')]
    
    total_mse_loss_3d = 0.0
    num_mse_loss_3d = 0

    for f in axial_volume_list:
        img_idx = f.split('.')[0].split('_')[1]
        file_path = os.path.join(data_dir, f)
        axial_volume = torch.tensor(load_tiff_stack(file_path).astype(np.float32) / 255.0, device=device)
        
        label_3d = label_dataset.get_original_label_3d(int(img_idx))
        label_3d_np = label_3d.squeeze()  # 去掉多余的维度
        label_3d_np = label_3d_np.transpose(1, 0, 2)
        label_3d_resized = resize_volume(label_3d_np, (150, 150, 150))
        coronal_reconstructed_volume = torch.tensor(label_3d_resized.astype(np.float32) / 255.0, device=device)

        mse_loss_3d = compute_3d_mse_loss(axial_volume, coronal_reconstructed_volume)
        total_mse_loss_3d = mse_loss_3d.item() + total_mse_loss_3d
        num_mse_loss_3d = num_mse_loss_3d + 1

    return total_mse_loss_3d / num_mse_loss_3d if num_mse_loss_3d > 0 else 0


def evaluate_two_volumes(pred, label, metrices=['dice', 'iou'], threshould=0.5, kernel:int=3, device=None):
    
    metric_func = {
    'iou': iou,
    'tiou': t_iou,
    'dice': dice_error
    }

    results_two = {}

    pred[pred < threshould] = 0
    pred[pred >= threshould] = 1
    label[label > 0] = 1

    pred = pred.astype(np.uint8)
    label = label.astype(np.uint8)

    pred = torch.Tensor(pred).view((1, 1, *pred.shape)).to(device)
    # pre = torch_dilation(pre, 5)
    label = torch.Tensor(label).view((1, 1, *label.shape)).to(device)

    pred = torch.nn.functional.max_pool3d(pred, kernel, 1, 0)
    label = torch.nn.functional.max_pool3d(label, kernel, 1, 0)

    for m in metrices:
        results_two[m] = metric_func[m](pred, label)
    
    return results_two


def evaluate_metric(data_dir:str, epoch:int, label_dataset, device,  preds_type='axial', metrics=['dice'], threshold=0.5, kernel=3, **kwargs):

    assert os.path.exists(data_dir)
    data_dir = os.path.join(data_dir, 'Epoch_'+str(epoch))
    
    assert os.path.exists(data_dir)

    results = {}
    for i in metrics:
        results[i] = []

    files = os.listdir(data_dir)
    axial_volume_list = [f for f in files if f.split('.')[0].endswith(preds_type)]
    
    total_mse_loss_3d = 0.0
    num_mse_loss_3d = 0

    for f in tqdm(axial_volume_list, desc=f'evaluating epoch {epoch}...'):
        img_idx = f.split('.')[0].split('_')[1]
        file_path = os.path.join(data_dir, f)
        axial_volume = load_tiff_stack(file_path).astype(np.uint8)
        
        label_3d = label_dataset.get_original_label_3d(int(img_idx))
        label_3d_np = label_3d.squeeze()  # 去掉多余的维度

        if preds_type == 'coronal':
            label_3d_np = label_3d_np.transpose(1, 0, 2)
        label_3d_resized = resize_volume(label_3d_np, (150, 150, 150))
        coronal_reconstructed_volume = label_3d_resized.astype(np.uint8)

        # mse_loss_3d = compute_3d_mse_loss(axial_volume, coronal_reconstructed_volume)
        # total_mse_loss_3d = mse_loss_3d.item() + total_mse_loss_3d
        # num_mse_loss_3d = num_mse_loss_3d + 1

        #*计算评价指标
        for m in metrics:
            res = evaluate_two_volumes(axial_volume, coronal_reconstructed_volume, metrices=metrics, threshould=threshold, device=device, kernel=kernel)
            for i in res.keys():
                results[i].append(res[i].item())
        

    
    # results['mse_loss'] = total_mse_loss_3d / num_mse_loss_3d if num_mse_loss_3d > 0 else 0

    for m in metrics:
        results[m] = np.array(results[m]).mean()

    return results

        
if __name__ == "__main__":

    data_dir = '/mnt/40B2A1DBB2A1D5A6/lyx/project/MedSam2/SAM-Unet/MODEL/model_trained/SAM2-Unet-GPS/正确的metric_2025_03_14_16:44/Samples'
    epoch = 0
    device = torch.device('cuda',0)

    args = config.dataset
    dataset_1 = FullDataset(size=352, mode='train', view='axial', **args)
    dataloader_1 = DataLoader(dataset_1, batch_size=args.batch_size, shuffle=False, num_workers=8)

    args = config.evaluate
    
    evaluate_metric(data_dir, epoch, dataset_1, device, **args)