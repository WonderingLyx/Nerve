from myutils import *
import os
import torch
import numpy as np
import argparse


def eval_two_volumes_maxpool(target, root, pool_kernel, device):
    """
    计算单个标签和预测的评估指标
    """
    label = read_tiff_stack(target)  # 读取标签
    pre = read_tiff_stack(root)  # 读取预测

    k = pool_kernel
    kernel = (k, k, k)
    pre[pre < 125] = 0
    pre[pre >= 125] = 1
    label[label > 0] = 1
    pre = pre.astype(np.uint8)
    label = label.astype(np.uint8)

    pre = torch.Tensor(pre).view((1, 1, *pre.shape)).to(device)
    label = torch.Tensor(label).view((1, 1, *label.shape)).to(device)

    pre = torch.nn.functional.max_pool3d(pre, kernel, 1, 0)
    label = torch.nn.functional.max_pool3d(label, kernel, 1, 0)

    dice_score = dice_error(pre, label)
    total_loss_iou = iou(pre, label).cpu()
    total_loss_tiou = t_iou(pre, label).cpu()
    clrecall, clprecision, recall, precision = soft_cldice_f1(pre, label)
    cldice = (2. * clrecall * clprecision) / (clrecall + clprecision)

    print('\nValidation IOU: {}\nT-IOU: {}\nClDice: {}\nClPrecision: {}\nClRecall: {}\nDice-score: {}\nPrecision: {}\nRecall: {}'
          .format(total_loss_iou, total_loss_tiou, cldice, clprecision, clrecall, dice_score, precision, recall))

    return {
        'iou': total_loss_iou,
        'tiou': total_loss_tiou,
        'cldice': cldice,
        'clprecision': clprecision,
        'clrecall': clrecall,
        'dice': dice_score,
        'precision': precision,
        'recall': recall
    }

def eval_all_volumes(target_dir, root_dir, pool_kernel, device):
    """
    遍历标签和预测文件夹，按文件名前缀匹配进行评估
    """
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.tif')])
    root_files = sorted([f for f in os.listdir(root_dir) if f.endswith('tif_prediction.tiff')])

    metrics_sum = {
        'iou': 0, 'tiou': 0, 'cldice': 0, 'clprecision': 0, 'clrecall': 0,
        'dice': 0, 'precision': 0, 'recall': 0
    }
    count = 0

    for target_file in target_files:
        # 获取文件名前缀，例如 i.tif 中的 "i"
        target_prefix = os.path.splitext(target_file)[0]

        # 在预测文件中查找对应文件
        root_file = f"{target_prefix}.tif_prediction.tiff"
        target_path = os.path.join(target_dir, target_file)
        root_path = os.path.join(root_dir, root_file)

        if os.path.exists(root_path):
            print(f"Processing: {target_file} and {root_file}")
            metrics = eval_two_volumes_maxpool(target_path, root_path, pool_kernel, device)

            # 累加指标
            for key in metrics_sum:
                metrics_sum[key] += metrics[key]
            count += 1
        else:
            print(f"Warning: No matching prediction file found for {target_file}")

    # 计算平均值
    if count > 0:
        avg_metrics = {key: val / count for key, val in metrics_sum.items()}
    else:
        avg_metrics = {key: 0 for key in metrics_sum}

    print("\nAverage Metrics:")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")

    return avg_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='/mnt/40B2A1DBB2A1D5A6/fxj/SAM2-UNet3/data/Task2641_P28/jinbiaozhun/tif/masks_8bit', help='Path to the directory containing ground-truth .tif files')
    parser.add_argument('--root', type=str, default='/mnt/40B2A1DBB2A1D5A6/fxj/SAM2-UNet3/output/multi/test/y-z/P28/subset/train/3D/cldice+150T+3Dmse+batch2loss2_60_4_0.0005', help='Path to the directory containing predicted .tiff files')
    parser.add_argument('--kernel_size', type=int, default=3, help='Maxpooling kernel size')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the evaluation (e.g., cuda:0 or cpu)')

    args = parser.parse_args()
    device = torch.device(args.device)

    # 批量评估
    eval_all_volumes(args.target, args.root, args.kernel_size, device)
