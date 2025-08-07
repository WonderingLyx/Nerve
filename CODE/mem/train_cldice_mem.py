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

config_path = '/mnt/40B2A1DBB2A1D5A6/lyx/project/MedSam2/SAM-Unet/CONFIG/SAM2-Unet_Mem.yaml'
config = OmegaConf.load(config_path)
args = config.general
config.save_dict = util.create_save_dir(args.save_path, args.project, args.exp_name, args.clear_all_model_save)

# #* 初始化WandB
# wandb.init(project=args.project, 
#            name= args.exp_name,
#            notes= args.notes,
#            dir= config.save_dict['log_path'],
#            config= OmegaConf.to_container(config)
#            )

# #* 设置不同的 wandb 横轴
# wandb.define_metric('train/epoch')
# wandb.define_metric('epoch/*', step_metric='train/epoch')


# 2D 损失函数 - structure_loss
def structure_loss(pred, mask):
    epsilon = 1e-7  # 避免分母为 0 的情况
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + epsilon)
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1 + epsilon)
    return (wbce + wiou).mean()

def compute_3d_mse_loss(axial_volume, coronal_reconstructed_volume):
    diff1 = axial_volume - coronal_reconstructed_volume
    diff2 = coronal_reconstructed_volume - axial_volume
    mse_loss = (torch.mean(diff1 ** 2) + torch.mean(diff2 ** 2)) / 2.0
    return mse_loss

# 3D 图像重采样
def resize_volume(volume, target_shape):
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.ndim} dimensions")
    depth_factor = target_shape[0] / volume.shape[0]
    height_factor = target_shape[1] / volume.shape[1]
    width_factor = target_shape[2] / volume.shape[2]
    resized = zoom(volume, (depth_factor, height_factor, width_factor), order=3)
    return resized

# 从 2D 切片重建 3D 图像
def reconstruct_3d_from_slices(input_dir, axis=0):
    slice_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.tiff') or f.endswith('.tif')])
    if not slice_files:
        raise ValueError(f"No valid slice files found in directory: {input_dir}")
    slices_2d = []
    for f in slice_files:
        slice_img = tiff.imread(os.path.join(input_dir, f))
        slices_2d.append(slice_img)
    reconstructed_3d = np.stack(slices_2d, axis=axis)
    return reconstructed_3d.astype(np.uint8)


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
    dataset_1 = FullDataset(size=352, mode='train', view='axial', **args)
    dataset_2 = FullDataset(size=352, mode='train', view='coronal', **args)
    dataloader_1 = DataLoader(dataset_1, batch_size=args.batch_size, shuffle=False, num_workers=8)
    dataloader_2 = DataLoader(dataset_2, batch_size=args.batch_size, shuffle=False, num_workers=8)

    print(f'the length of dataset axial is: {len(dataset_1)}')
    #print(f'the length of dataset coronal is: {len(dataset_2)}')
    
    device = torch.device("cuda")

    args = config.model
    model_1 = SAM2UNet(args.sam_type, args.hiera_path, args.args).to(device)
    #model_2 = SAM2UNet(args.sam_type, args.hiera_path, args.args).to(device)

    print(f'model = {str(model_1)}')

    #* 学习率
    num_training_steps_per_epoch = len(dataset_1) // config.dataset.batch_size
    args = config.lr_schedule
    if args.is_use:
        lr_schedule_values = cosine_scheduler(
            epochs=config.train.epoch, niter_per_ep=num_training_steps_per_epoch, **args
        )
    else:
        lr_schedule_values = None

    args = config.train
    optim_1 = opt.AdamW(model_1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optim_2 = opt.AdamW(model_2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_1 = CosineAnnealingLR(optim_1, T_max=args.epoch, eta_min=1e-7)
    #scheduler_2 = CosineAnnealingLR(optim_2, T_max=args.epoch, eta_min=1e-7)
    lambda_3d = args.lambda_3d


    #* Training
    for epoch in range(args.epoch):
        model_1.train()
        #model_2.train()
        total_loss_1, total_loss_2 = 0.0, 0.0
        total_cldice_axial, total_tiou_axial, total_iou_axial = 0.0, 0.0, 0.0
        step, num_cldice_axial = 0, 0
         # 聚合切片预测结果
        pred_slices_dict = {}
        
        batch_id = 0
        for batch_1, batch_2 in zip(tqdm(dataloader_1,desc=f'Training Process@ Epoch {epoch}...'), dataloader_2):
            x_1, target_1 = batch_1['image'].to(device), batch_1['label'].to(device)
            #x_2, target_2 = batch_2['image'].to(device), batch_2['label'].to(device)
            image_index = batch_1['image_index'][0]
            # Axial 模型训练
            optim_1.zero_grad()

            #* update lr
            if lr_schedule_values is not None:
                it = epoch * num_training_steps_per_epoch + step
                for i, param_group in enumerate(optim_1.param_groups):
                    param_group['lr'] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
            
            pred_1, pred1_aux1, pred1_aux2 = model_1(x_1, image_index, batch_id, config.dataset.batch_size)
            loss_1_main = structure_loss(pred_1, target_1)
            loss_1_aux1 = structure_loss(pred1_aux1, target_1)
            loss_1_aux2 = structure_loss(pred1_aux2, target_1)
            loss_1 = loss_1_main+loss_1_aux1 + loss_1_aux2
            loss_1.backward()
            torch.nn.utils.clip_grad_norm_(model_1.parameters(), max_norm=1.0)
            optim_1.step()
            total_loss_1 = total_loss_1 + loss_1.item()

            if torch.isnan(loss_1) or torch.isinf(loss_1):
                print(f"NaN or Inf loss detected in Axial model at epoch {epoch + 1}")
                return
            # # Coronal 模型训练
            # optim_2.zero_grad()
            # pred_2, pred2_aux1, pred2_aux2 = model_2(x_2)
            # loss_2_main = structure_loss(pred_2, target_2)
            # loss_2_aux1 = structure_loss(pred2_aux1, target_2)
            # loss_2_aux2 = structure_loss(pred2_aux2, target_2)
            # loss_2 = loss_2_main + 0.5 * (loss_2_aux1 + loss_2_aux2)
            # loss_2.backward()
            # optim_2.step()
            # total_loss_2 += loss_2.item()
            min_lr = 10.
            max_lr = 0.
            for group in optim_1.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            wandb.log({
                'Axial/loss_per_batch': loss_1.item(),
                'Axial/min_lr': min_lr,
                'Axial/max_lr': max_lr
                #'Coronal/loss_per_batch': loss_2.item()
            })

            # 将切片预测结果加入字典
            for idx, image_index in enumerate(batch_1['image_index']):
                if image_index.item() not in pred_slices_dict:
                    pred_slices_dict[image_index.item()] = []
                pred_slices_dict[image_index.item()].append(pred_1[idx])

            step = step + 1
            batch_id = batch_id + 1

        #* evaluate train
        for image_index in pred_slices_dict.keys():
            pred_slices_1 = pred_slices_dict[image_index]

            # 获取完整的3D图像和标签
            original_3d_1 = dataset_1.get_original_3d(image_index)
            #original_3d_2 = dataset_2.get_original_3d(image_index)
            label_3d_1 = dataset_1.get_original_label_3d(image_index)
            #label_3d_2 = dataset_2.get_original_label_3d(image_index)

            # 保存 Axial 视角的 3D 预测
            output_path = os.path.join(config.save_dict['sample_path'], f'epoch_{epoch}')

            pred_file_axial, label_file_axial, _ = save_3d_prediction_and_label(
                pred_slices_1, original_3d_1, label_3d_1, epoch, view="axial",
                output_path=output_path, name=f"Image_{image_index}_Axial"
            )
            results_axial = eval_two_volumes_maxpool(pred_file_axial, label_file_axial, pool_kernel=2, device=device)
            print(f"Axial View - Epoch {epoch + 1}, Image {image_index}: ClDice = {results_axial['cldice']:.4f}")
            total_cldice_axial += results_axial['cldice']
            total_tiou_axial += results_axial['tiou']
            total_iou_axial += results_axial['iou']
            num_cldice_axial += 1


        #avg_cldice_axial = total_cldice_axial / num_cldice_axial if num_cldice_axial > 0 else 0
        # save_predics(dataloader_1, model_1, device, epoch, 'axial', config.save_dict['sample_path'])
        # #save_predics(dataloader_2, model_2, device, epoch, 'coronal', config.save_dict['sample_path'])
        # args = config.evaluate
        # results = evaluate_metric(config.save_dict['sample_path'], epoch, dataset_1, device, **args)
        
        wandb.log({
#                'Epoch/mse_loss': results['mse_loss'],
            'Epoch/iou': total_iou_axial / num_cldice_axial,
            'Epoch/tiou': total_tiou_axial / num_cldice_axial,
            'Epoch/dice': total_cldice_axial / num_cldice_axial
        })
        # 更新学习率调度器
        scheduler_1.step()
        #scheduler_2.step()

        # 计算平均指标
        avg_loss_1 = total_loss_1 / len(dataloader_1) if len(dataloader_1) > 0 else 0
        # avg_loss_2 = total_loss_2 / len(dataloader_2) if len(dataloader_2) > 0 else 0

        # 计算总损失
        # if epoch >= 5:
        #     total_loss = avg_loss_1 + avg_loss_2 + lambda_3d * results['mse_loss']
        # else:
        #     total_loss = avg_loss_1 + avg_loss_2

        #打印日志
        wandb.log({
            'train/epoch': epoch,
            'Epoch/Avg Loss_1': avg_loss_1,
            #'Epoch/Avg Loss_2': avg_loss_2,
            #'Epoch/Avg MSE 3D Loss': avg_mse_loss_3d,
        })

        # 保存模型
        if (epoch+1) % 5 == 0 or (epoch + 1) == config.train.epoch:
            torch.save(model_1.state_dict(), os.path.join(config.save_dict['ckpt_path'], f"SAM2-UNet-view1-epoch-{epoch + 1}.pth"))
            #torch.save(model_2.state_dict(), os.path.join(config.save_dict['ckpt_path'], f"SAM2-UNet-view2-epoch-{epoch + 1}.pth"))
            print(f"Saved model checkpoints for epoch {epoch + 1}.")

if __name__ == "__main__":
    main()