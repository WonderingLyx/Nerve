import os
import argparse
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from SAM2UNet import SAM2UNet
import wandb  # 用于日志记录
import tifffile as tiff  # 用于保存3D图像
from PIL import Image  # 用于保存 2D 切片
from eval_volumes import eval_two_volumes_maxpool  # 调用标准 ClDice 计算脚本
from scipy.ndimage import zoom  # 用于 3D 图像的重采样
import subprocess
from reconstruct2 import load_tiff_stack, reconstruct_original_from_coronal, apply_transformations, save_tiff_stack
from omegaconf import OmegaConf
from Myutils import util
from tqdm import tqdm
from evaluate import save_predics, evaluate_metric
from utils import cosine_scheduler

config_path = '/mnt/40B2A1DBB2A1D5A6/lyx/project/MedSam2/SAM-Unet/CONFIG/SAM2-Unet_GPS_pred.yaml'
config = OmegaConf.load(config_path)
args = config.general
#config.save_dict = util.create_save_dir(args.save_path, args.project, args.exp_name, args.clear_all_model_save)

# ###* 初始化WandB
# wandb.init(project=args.project, 
#            name= args.exp_name,
#            notes= args.notes,
#            #dir= config.save_dict['log_path'],
#            config= OmegaConf.to_container(config)
#            )

def resize_volume(volume, target_shape):
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.ndim} dimensions")
    depth_factor = target_shape[0] / volume.shape[0]
    height_factor = target_shape[1] / volume.shape[1]
    width_factor = target_shape[2] / volume.shape[2]
    resized = zoom(volume, (depth_factor, height_factor, width_factor), order=3)
    return resized

def save_3d_prediction_and_label(pred_slices, original_3d, label_3d, epoch, view, output_path, name):
    os.makedirs(output_path, exist_ok=True)

    # 1. 保存 3D 预测结果
    pred_3d = torch.stack([torch.sigmoid(pred).detach().cpu() for pred in pred_slices])  # Shape: [depth, height, width]
    pred_3d_np = pred_3d.numpy().squeeze()  # 去掉多余的维度 ([depth, height, width])

    if pred_3d_np.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {pred_3d_np.shape}")

    # Resize to (150, 150, 150)
    pred_3d_resized = resize_volume(pred_3d_np, (150, 150, 150))
    pred_3d_resized = (pred_3d_resized * 255).astype(np.uint8)
    pred_filename = os.path.join(output_path, f"{name}_view_{view}_prediction.tiff")
    tiff.imwrite(pred_filename, pred_3d_resized, photometric='minisblack')
    #print(f"3D prediction saved for view {view} at epoch {epoch} as {pred_filename}")

    # 初始化 reconstructed_filename
    reconstructed_filename = None

    # 如果是 Coronal 视角，执行重建操作
    if view == "coronal":
        reconstructed_filename = os.path.join(output_path, f"{name}_epoch_{epoch}_view_{view}_reconstructed_prediction.tiff")
        original_shape = (150, 150, 150)  # 原始形状 (depth, height, width)

        # 调用重建脚本逻辑
        stacked_image = load_tiff_stack(pred_filename)  # 读取保存的冠状面预测
        reconstructed_image = reconstruct_original_from_coronal(stacked_image, original_shape)  # 重建
        final_image = apply_transformations(reconstructed_image)  # 应用旋转、翻转变换

        # 保存重建后的 3D 图像
        save_tiff_stack(final_image, reconstructed_filename)
        #print(f"Reconstructed 3D prediction saved for view {view} at epoch {epoch} as {reconstructed_filename}")

    # 2. 保存原始 3D 图像
    # 此处不再调用 .cpu()，因为 original_3d 已是 numpy 数组
    original_3d_np = original_3d.squeeze()  # 确保形状正确
    if view == "coronal":
        original_3d_np = original_3d_np.transpose(1, 0, 2)

    if original_3d_np.ndim != 3:
        raise ValueError(f"Expected 3D array for original image, got shape {original_3d_np.shape}")

    original_3d_resized = resize_volume(original_3d_np, (150, 150, 150))
    original_filename = os.path.join(output_path, f"{name}_view_{view}_original.tiff")
    tiff.imwrite(original_filename, original_3d_resized.astype(np.uint8), photometric='minisblack')
    #print(f"3D original image saved for view {view} at epoch {epoch} as {original_filename}")

    # 3. 保存标签
    # 此处同样不再调用 .cpu()，因为 label_3d 也是 numpy 数组
    label_3d_np = label_3d.squeeze()
    if view == "coronal":
        label_3d_np = label_3d_np.transpose(1, 0, 2)

    if label_3d_np.ndim != 3:
        raise ValueError(f"Expected 3D array for label, got shape {label_3d_np.shape}")

    label_3d_resized = resize_volume(label_3d_np, (150, 150, 150))
    label_filename = os.path.join(output_path, f"{name}_view_{view}_label.tiff")
    tiff.imwrite(label_filename, label_3d_resized, photometric='minisblack')
    #print(f"3D label saved for view {view} at epoch {epoch} as {label_filename}")

    # 返回文件路径：Axial 不包含 reconstructed_filename
    return pred_filename, label_filename, reconstructed_filename


def main():

    args = config.dataset
    dataset_1 = FullDataset(size=352, mode='val', view='axial', **args)
    # dataset_2 = FullDataset(size=352, mode='train', view='coronal', **args)
    dataloader_1 = DataLoader(dataset_1, batch_size=args.batch_size, shuffle=False, num_workers=8)
    # dataloader_2 = DataLoader(dataset_2, batch_size=args.batch_size, shuffle=False, num_workers=8)

    print(f'the length of dataset axial is: {len(dataset_1)}')
    #print(f'the length of dataset coronal is: {len(dataset_2)}')
    
    device = torch.device("cuda")

    args = config.model
    ckpt_path = args.ckpt_path

    model_1 = SAM2UNet(args.sam_type, args.hiera_path, args.args).to(device)
    if os.path.exists(ckpt_path):
        
        # save_path = os.path.dirname(os.path.dirname(ckpt_path))
        # save_path = os.path.join(save_path, 'Eval')
        # os.makedirs(save_path, exist_ok=True)
        save_path = args.save_path
        ckpt = torch.load(ckpt_path, map_location=device)
        model_1.load_state_dict(ckpt)

    else:
        return f'Invalid ckpt path'

    # model_1.eval()
    pred_slices_dict = {}
    #* Prediction
    for batch_id, batch in enumerate(tqdm(dataloader_1, desc=f'Predicting...')):
        x, target = batch['image'].to(device), batch['label'].to(device)

        image_index = batch['image_index'][0]

        with torch.no_grad():
            pred_1, pred1_aux1, pred1_aux2 = model_1(x)

        for idx, image_index in enumerate(batch['image_index']):
            if image_index.item() not in pred_slices_dict:
                pred_slices_dict[image_index.item()] = []
            pred_slices_dict[image_index.item()].append(pred_1[idx])
    
    total_cldice_axial = 0.0
    total_tiou_axial = 0.0
    total_iou_axial = 0.0
    num_cldice_axial = 0
    #* evaluation 
    for image_index in pred_slices_dict.keys():
        pred_slices_1 = pred_slices_dict[image_index]

        # 获取完整的3D图像和标签
        original_3d_1 = dataset_1.get_original_3d(image_index)
        #original_3d_2 = dataset_2.get_original_3d(image_index)
        label_3d_1 = dataset_1.get_original_label_3d(image_index)
        #label_3d_2 = dataset_2.get_original_label_3d(image_index)
        name = dataset_1.get_file_name(image_index)
        # 保存 Axial 视角的 3D 预测

        pred_file_axial, label_file_axial, _ = save_3d_prediction_and_label(
            pred_slices_1, original_3d_1, label_3d_1, 0, view="axial",
            output_path=save_path, name=f"Image_{name}_Axial"
        )
        results_axial = eval_two_volumes_maxpool(pred_file_axial, label_file_axial, pool_kernel=2, device=device)
        print(f"Axial View - Epoch {config.model.train_epoch + 1}, Image {name}: ClDice = {results_axial['cldice']:.4f}")
        total_cldice_axial += results_axial['cldice']
        total_tiou_axial += results_axial['tiou']
        total_iou_axial += results_axial['iou']
        num_cldice_axial += 1

    
    print(f'ave_cldice_axial: {(total_cldice_axial / num_cldice_axial):.4f}')
    print(f'ave_tiou_axial: {(total_tiou_axial / num_cldice_axial):.4f}')
    print(f'ave_iou_axial: {(total_iou_axial / num_cldice_axial):.4f}')

    



if __name__ == "__main__":
    main()