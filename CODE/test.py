import os
import argparse
import torch
from dataset import TestDataset
from SAM2UNetDC import SAM2UNet
import tifffile as tiff
import numpy as np
from scipy.ndimage import zoom
from PIL import Image

def resize_volume(volume, target_shape):
    """
    3D 图像重采样到指定形状。
    :param volume: 输入 3D 图像
    :param target_shape: 目标形状 (depth, height, width)
    :return: 重采样后的 3D 图像
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.ndim} dimensions")

    # 确保 target_shape 是普通整数而非 Tensor
    target_shape = [int(dim) for dim in target_shape]

    # 计算缩放比例
    depth_factor = target_shape[0] / volume.shape[0]
    height_factor = target_shape[1] / volume.shape[1]
    width_factor = target_shape[2] / volume.shape[2]
    resized = zoom(volume, (depth_factor, height_factor, width_factor), order=3)  # 使用三次插值
    return resized


def save_3d_prediction(pred_slices, original_shape, output_path, name):
    """
    保存 3D 预测图像，调整为与原始尺寸一致。
    :param pred_slices: 模型预测的 2D 切片列表
    :param original_shape: 原始 3D 图像形状
    :param output_path: 保存路径
    :param name: 文件名
    """
    os.makedirs(output_path, exist_ok=True)

    # 合并所有切片并转换为 NumPy 格式
    pred_3d = torch.stack([torch.sigmoid(pred).detach().cpu() for pred in pred_slices])  # Shape: [depth, height, width]
    pred_3d_np = pred_3d.numpy().squeeze()

    if pred_3d_np.ndim != 3:
        raise ValueError(f"[ERROR] Expected 3D array, got shape {pred_3d_np.shape}")

    # 调整为原始形状
    pred_3d_resized = resize_volume(pred_3d_np, original_shape)
    pred_3d_resized = (pred_3d_resized * 255).astype(np.uint8)

    # 保存为 tiff 文件
    pred_filename = os.path.join(output_path, f"{name}_prediction.tiff")
    tiff.imwrite(pred_filename, pred_3d_resized, photometric='minisblack')
    print(f"[INFO] 3D prediction saved at: {pred_filename}")

'''
def save_original_slices(original_slices, output_dir, base_name):
    """
    保存原始的2D切片。
    :param original_slices: 原始切片列表（每个切片应为 NumPy 数组）。
    :param output_dir: 保存目录。
    :param base_name: 基础文件名。
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, slice_2d in enumerate(original_slices):
        if isinstance(slice_2d, torch.Tensor):
            # 如果是 Tensor，则先转换为 NumPy 数组
            slice_2d = slice_2d.cpu().numpy()

        # 移除多余的维度（如果有）
        slice_2d = np.squeeze(slice_2d)

        # 检查是否是 2D 数据
        if slice_2d.ndim != 2:
            raise ValueError(f"[ERROR] Slice {i + 1} is not 2D after squeezing, shape: {slice_2d.shape}")

        img = Image.fromarray(slice_2d.astype(np.uint8))  # 转换为 PIL 图像
        slice_filename = os.path.join(output_dir, f"{base_name}_slice_{i + 1}.png")
        img.save(slice_filename)

        print(f"[INFO] Saved original slice: {slice_filename}")
'''

def test(args):

    device = torch.device(args.device_test if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 加载测试集
    test_dataset = TestDataset(
        image_root=args.test_image_path,
        size=352,
        view='axial',
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 加载模型
    model = SAM2UNet(
        model_cfg='sam2_hiera_l',
        checkpoint_path=args.pretrain_path,
        args=args,
        device=device
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    if torch.cuda.is_available():
        model.to(device=device)  
    model.eval()
    print(f"[INFO] Loaded model from checkpoint: {args.checkpoint}")

    # 遍历测试数据
    for index, (slices, original_shape, name, original_slices) in enumerate(test_loader):
        print(f"[INFO] Processing 3D image {index + 1}/{len(test_loader)}: {name[0]}")

        # 确保 original_shape 是普通 Python 类型
        original_shape = [int(dim) for dim in original_shape]  # 转换为普通整数
        print(f"[DEBUG] Original 3D shape: {original_shape}")

        pred_slices = []

        # 遍历所有切片并进行推理
        for i, slice_2d in enumerate(slices[0]):  # slices[0] 是切片
            slice_2d = slice_2d.unsqueeze(0).to(device)  # [1, 3, H, W]

            # 模型推理
            with torch.no_grad():
                pred, _, _ = model(slice_2d)  # 推理
                pred = pred.squeeze()  # [H, W]
                pred_slices.append(pred)

            # 打印调试信息
            if i % 10 == 0 or i == slices.shape[1] - 1:
                print(f"[DEBUG] Predicted slice {i + 1}/{slices.shape[1]}, shape: {pred.shape}")

        # 确保切片数量正确
        if len(pred_slices) != original_shape[0]:  # 检查切片数是否匹配 depth
            raise ValueError(f"[ERROR] Predicted slices contain {len(pred_slices)} slices, expected {original_shape[0]} slices.")

        print(f"[INFO] Total predicted slices: {len(pred_slices)}")

        # 保存最终的 3D 预测结果
        save_3d_prediction(
            pred_slices=pred_slices,
            original_shape=original_shape,
            output_path=args.predict_path,
            name=name[0]  # 使用图像的名称作为文件名
        )

        # 保存原始的2D切片
#        original_slices_dir = os.path.join(args.output_path, "original_slices")
#        save_original_slices(original_slices, original_slices_dir, name[0])


