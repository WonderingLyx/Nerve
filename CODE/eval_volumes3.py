from myutils import *
import os
import torch
import numpy as np
import argparse
import cripser
from skimage import measure
from functools import reduce
import gudhi

indx_dir = '/root/shared-nvme/lyx/Project/Nerve/Trial/Nerve-sam2post/CONFIG'
axis = ['x', 'y', 'z']
patch_indx = []
for k in axis:
    indx_path = os.path.join(indx_dir, f'index_{k}.npy')
    patch_indx.append(np.load(indx_path).tolist())

patch_indx = np.array(patch_indx)


def ramdon_cut(vol, lb, patch_size=None, indexs=None):
    if len(patch_size) == 3:
        x_max, y_max, z_max = vol.shape

        if indexs is None:
            x_idx = np.random.randint(0, x_max - patch_size[0])
            y_idx = np.random.randint(0, y_max - patch_size[1])
            z_idx = np.random.randint(0, z_max - patch_size[2])
        
        else:
            (x_idx, y_idx, z_idx) = indexs

        vol_cut = vol[x_idx:x_idx + patch_size[0], y_idx:y_idx+patch_size[0], z_idx:z_idx+patch_size[2]]
        lb_cut = lb[x_idx:x_idx + patch_size[0], y_idx:y_idx+patch_size[0], z_idx:z_idx+patch_size[2]]
    
    if len(patch_size) == 2:
        x_max, y_max = vol.shape

        if indexs is None:
            x_idx = np.random.randint(0, x_max - patch_size[0])
            y_idx = np.random.randint(0, y_max - patch_size[1])
        
        else:
            (x_idx, y_idx) = indexs

        vol_cut = vol[x_idx:x_idx + patch_size[0], y_idx:y_idx+patch_size[0]]
        lb_cut = lb[x_idx:x_idx + patch_size[0], y_idx:y_idx+patch_size[0]]

    
    return vol_cut, lb_cut
    
def betti_csp(preds_cut, labels_cut, noise=1e-5):
    betti_pre = {0: 0, 1: 0, 2:0}
    betti_lb = {0: 0, 1: 0, 2:0}

    pred_res = cripser.computePH(preds_cut, maxdim=2)
    labels_res = cripser.computePH(labels_cut, maxdim=2)

    for diagram in pred_res:
        dim = diagram[0]  # 同调维度（0,1,2）
        birth = diagram[1]  # 出生时间
        death = diagram[2]  # 死亡时间
        if death - birth > noise:  # 过滤掉短暂存在的噪声特征
            betti_pre[dim] += 1
    
    for diagram in labels_res:
        dim = diagram[0]  # 同调维度（0,1,2）
        birth = diagram[1]  # 出生时间
        death = diagram[2]  # 死亡时间
        if death - birth > noise:  # 过滤掉短暂存在的噪声特征
            betti_lb[dim] += 1
    
    return betti_pre, betti_lb

def compute_betti_gud(matrix,min_pers=0,i=5):

    """
        Given a matrix representing a nii image compute the persistence diagram by using the Gudhi library (link)

        :param matrix: matrix encoding the nii image
        :type matrix: np.array

        :param min_pers: minimum persistence interval to be included in the persistence diagram
        :type min_pers: Integer

        :returns: Persistence diagram encoded as a list of tuples [d,x,y]=p where

            * d: indicates the dimension of the d-cycle p

            * x: indicates the birth of p

            * y: indicates the death of p
    """
    #save the dimenions of the matrix
    dims = matrix.shape
    size = reduce(lambda x, y: x * y, dims, 1)

    #create the cubica complex from the image
    cubical_complex = gudhi.CubicalComplex(dimensions=dims,top_dimensional_cells=np.reshape(matrix.T,size))
    #compute the persistence diagram
    if i == 5:
        pd = cubical_complex.persistence(homology_coeff_field=2, min_persistence=min_pers)
        return np.array(map(lambda row: [row[1][0],row[1][1]], pd))
    else:
        pd = cubical_complex.persistence(homology_coeff_field=2, min_persistence=min_pers)
        pd = cubical_complex.persistence_intervals_in_dimension(i)
        pd = np.array(list(map(lambda row: [row[0],row[1]], pd)))

        return len(pd)
    
def betti_gud(pred, label):
    betti_pre = {0: 0, 1: 0, 2:0}
    betti_lb = {0: 0, 1: 0, 2:0}

    for i in range(3):
        pre_x = compute_betti_gud(pred, i=i)
        betti_pre[i] = pre_x
    
    for i in range(3):
        lb_x = compute_betti_gud(label, i=i)
        betti_lb[i] = lb_x

    return betti_pre, betti_lb


def eval_two_volume_betti(preds, labels, noise=None, patch_size=(16,16,16), num=100):

    if noise == None:
        noise = 1e-5

    if preds.dtype is not np.float64:
        preds = preds.astype(np.float64) 
    
    if labels.dtype is not np.float64:
        labels = labels.astype(np.float64)

    beta = {
        0:0,
        1:0,
        2:0
    }

    for i in range(num):
        preds_cut, labels_cut = ramdon_cut(preds, labels, patch_size=patch_size, indexs=patch_indx[:,i])
        
        #TODO crisper
        betti_pre, betti_lb = betti_csp(preds_cut, labels_cut)
        
        #TODO gudhi 
        #betti_pre, betti_lb = betti_gud(preds_cut, labels_cut)

        #print(betti_lb[1])
        for i in range(3):
            beta[i] += (abs(betti_pre[i] - betti_lb[i]))
    
    for i in range(3):
        beta[i] = beta[i] / num
        #print(f'beta{i}: {beta[i]}')

    return beta

def eval_two_volume_Euler(preds, labels, connectivity=4, patch_size=(16,16,16), num=100):

    if preds.dtype is not np.uint8:
        preds = preds.astype(np.uint8) 
    
    if labels.dtype is not np.uint8:
        labels = labels.astype(np.uint8)

    euler_diff = 0

    for i in range(num):
        preds_cut, labels_cut = ramdon_cut(preds, labels, patch_size=patch_size, indexs=patch_indx[:,i])
        preds_euler = measure.euler_number(preds_cut, connectivity=connectivity)
        lbs_euler = measure.euler_number(labels_cut, connectivity=connectivity)

        euler_diff += abs(preds_euler  - lbs_euler)

        
    
    euler_diff = euler_diff / num

    return euler_diff


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

    pre = pre.cpu().numpy().squeeze()
    label = label.cpu().numpy().squeeze()

    bettis = eval_two_volume_betti(pre, label)

    eulers = eval_two_volume_Euler(pre, label)


    print('\nValidation IOU: {:.4f}\nT-IOU: {:.4f}\nClDice: {:.4f}\nClPrecision: {:.4f}\nClRecall: {:.4f}\nDice-score: {:.4f}\nPrecision: {:.4f}\nRecall: {:.4f}\nBetti0: {}\nBetti1: {}\nEuler: {}'
          .format(total_loss_iou, total_loss_tiou, cldice, clprecision, clrecall, dice_score, precision, recall, bettis[0], bettis[1], eulers))

    return {
        'iou': total_loss_iou,
        'tiou': total_loss_tiou,
        'cldice': cldice,
        'clprecision': clprecision,
        'clrecall': clrecall,
        'dice': dice_score,
        'precision': precision,
        'recall': recall,
        'betti0': bettis[0],
        'betti1': bettis[1],
        'euler': eulers
    }

def eval_all_volumes(target_dir, root_dir, pool_kernel, device):
    """
    遍历标签和预测文件夹，按文件名前缀匹配进行评估
    """
    #print(target_dir)
    #target_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.tiff')])
    
    target_files = [f for f in os.listdir(target_dir) if f.endswith('.tiff') or f.endswith('.tif')]
    target_files.sort(key= (lambda x: int(x.split('.')[0].split('-')[-1])))

    root_files = sorted([f for f in os.listdir(root_dir) if f.endswith('tif_prediction.tiff')])

    metrics_sum = {
        'iou': 0, 'tiou': 0, 'cldice': 0, 'clprecision': 0, 'clrecall': 0,
        'dice': 0, 'precision': 0, 'recall': 0, 'betti0': 0, 'betti1': 0, 'euler':0
    }
    count = 0

    for target_file in target_files:
        # 获取文件名前缀，例如 i.tif 中的 "i"
        print(target_file)
        target_prefix = os.path.splitext(target_file)[0]

        # 在预测文件中查找对应文件
        #TODO 可以修改对应后缀
        #*Origin
        #root_file = f"{target_prefix}.tiff_prediction.tiff"
        #*DC
        root_file = f"{target_prefix}-DC.tif"


        #print(root_file)
        target_path = os.path.join(target_dir, target_file)
        root_path = os.path.join(root_dir, root_file)

        if os.path.exists(root_path):
            print(f"Processing: {target_file}")
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
    parser.add_argument('--target', type=str, default='/root/shared-nvme/lyx/Project/Nerve/Data/Sert-Stanford/test/test/label-8bit', help='Path to the directory containing ground-truth .tif files')
    parser.add_argument('--root', type=str, default='/root/shared-nvme/lyx/Project/Nerve/Trial/Res/Predicts/Predicts_DC', help='Path to the directory containing predicted .tiff files')
    parser.add_argument('--kernel_size', type=int, default=3, help='Maxpooling kernel size')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the evaluation (e.g., cuda:0 or cpu)')

    args = parser.parse_args()
    device = torch.device(args.device)

    # 批量评估
    eval_all_volumes(args.target, args.root, args.kernel_size, device)
