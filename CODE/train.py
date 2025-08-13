import os
import argparse
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from SAM2UNetDC import SAM2UNet
import wandb
import tifffile as tiff
from reconstruct2 import load_tiff_stack, reconstruct_original_from_coronal, apply_transformations, save_tiff_stack
from scipy.ndimage import zoom
import logging
from datetime import datetime
import sys


# 定义2D的损失函数
def structure_loss(pred, mask):
    epsilon = 1e-7
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

# 定义3D图像重采样函数
def resize_volume(volume, target_shape):
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.ndim} dimensions")
    depth_factor = target_shape[0] / volume.shape[0]
    height_factor = target_shape[1] / volume.shape[1]
    width_factor = target_shape[2] / volume.shape[2]
    resized = zoom(volume, (depth_factor, height_factor, width_factor), order=3)
    return resized

# 保存和评估3D预测与标签
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
    pred_filename = os.path.join(output_path, f"{name}_epoch_{epoch}_view_{view}_prediction.tiff")
    tiff.imwrite(pred_filename, pred_3d_resized, photometric='minisblack')
    print(f"3D prediction saved for view {view} at epoch {epoch} as {pred_filename}")

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
        print(f"Reconstructed 3D prediction saved for view {view} at epoch {epoch} as {reconstructed_filename}")

    # 返回文件路径：Axial 不包含 reconstructed_filename
    return pred_filename, reconstructed_filename

def Get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    #os.makedirs(args.output_path, exist_ok=True)
    dt = datetime.today()
    log_name = (args.test_id + '_' + str(dt.date()) + "_" + str(dt.time().hour) + ":" +
                str(dt.time().minute) + ":" + str(dt.time().second) + "_" 
                )
    logger = Get_logger(args.save_path +'/Log/' + log_name+'.log')


    # 数据集和数据加载器
    dataset = FullDataset(image_root=args.train_image_path, gt_root=args.train_mask_path, size=352, mode='train', view='axial')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = SAM2UNet(
        model_cfg='sam2_hiera_l',
        checkpoint_path=args.pretrain_path,
        args=args,
        device=device
    )

    if torch.cuda.is_available():
        model.to(device=device)      

    optim = opt.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = CosineAnnealingLR(optim, T_max=args.epoch, eta_min=1e-7)

    logger.info("Start Training")
    for epoch in range(args.epoch):
        model.train()

        total_loss = 0.0
        total_mse_loss_3d = 0.0
        num_mse_loss_3d = 0

        # 聚合切片预测结果
        pred_slices_dict = {}

        for batch_id, batch in enumerate(dataloader):
            if torch.cuda.is_available():
                x, target = batch['image'].to(device), batch['label'].to(device)
            else:
                x, target = batch['image'], batch['label']

            # Axial 模型训练
            optim.zero_grad()
            pred, pred_aux1, pred_aux2 = model(x)
            loss_main = structure_loss(pred, target)
            loss_aux1 = structure_loss(pred_aux1, target)
            loss_aux2 = structure_loss(pred_aux2, target)
            loss = loss_main + 0.5 * (loss_aux1 + loss_aux2)
            loss.backward()

            #* 检查梯度
            grad_die = False
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any():
                        grad_die = True
                        logger.info(f'Epoch:{epoch} batch:{batch_id}, gradient is NAN')
                        sys.exit("gradient is NAN")
                    if torch.isinf(p.grad).any():
                        grad_die=True
                        logger.info(f'Epoch:{epoch} batch:{batch_id}, gradient is INF')
                        sys.exit("gradient is INF")
                    
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            
            total_norm = total_norm ** 0.5

            if grad_die:
                torch.save(model.state_dict(), f'{args.save_path}/Model/{args.test_id}-StopEpoch-{epoch}-StopIter-{batch_id}.pth')

            total_loss += loss.item()
            Loss = total_loss / (batch_id+1)
            logger.info(f"Epoch: {epoch} Iter:{batch_id} lr:{optim.param_groups[0]['lr']} Batch_loss:{loss.item():.6f} Loss:{Loss:6f} norm:{total_norm:4f}")

            optim.step()
            # 将切片预测结果加入字典
            # for idx, image_index in enumerate(batch['image_index']):
            #     if image_index.item() not in pred_slices_dict:
            #         pred_slices_dict[image_index.item()] = []
            #     pred_slices_dict[image_index.item()].append(pred[idx])


        # 处理聚合的切片
        # for image_index in pred_slices_dict.keys():
        #     pred_slices = pred_slices_dict[image_index]
            

        #     # 获取完整的3D图像和标签
        #     original_3d = dataset.get_original_3d(image_index)
            
        #     label_3d = dataset.get_original_label_3d(image_index)
            

            # 保存 Axial 视角的 3D 预测
            # pred_file_axial, _ = save_3d_prediction_and_label(
            #     pred_slices, original_3d, label_3d, epoch, view="axial",
            #     output_path=args.output_path, name=f"Image_{image_index}_Axial"
            # )
    

        # 更新学习率调度器
        scheduler.step()
        # 计算平均指标
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0

        # 打印日志
        logger.info(f"Epoch [{epoch + 1}/{args.epoch}] - "
              f"Avg Loss: {avg_loss:.4f} "
              )


        # 保存模型
        if (epoch + 1) % 1 == 0 or (epoch + 1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path + '/Model', f"{args.test_id}-epoch-{epoch + 1}.pth"))
            logger.info(f"Saved model checkpoints for epoch {epoch + 1}.")

