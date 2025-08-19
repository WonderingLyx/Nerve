import torch
import numpy as np
from monai.transforms import RandAffine, RandSpatialCrop
import random
import SimpleITK as sitk
import os
import tifffile as tiff

def mask_generator3D_v2(input_tensor, label_tensor, mask_ratios, patch_sizes, if_random_affine=False, threshold=0.3):
    """
    generate 3D mask with different ratios for different labels
    :param input_tensor: input tensor [B, 1, D, H, W]
    :param label_tensor: label tensor [B, 1, D, H, W] containing label values (e.g., 0 for background, 1 for foreground)
    :param mask_ratios: dictionary of mask ratios for each label, e.g., {0:0.2, 1:0.5}
    :param patch_sizes: the patch size of cubic mask [d, h, w]
    :param if_random_affine: whether to apply random rotation onto the generated 3D mask
    :return: mask with shape [B,1,D,H,W], 0 is to keep, 1 is to remove
    """
    # 验证输入参数
    assert type(patch_sizes) == tuple and len(patch_sizes) == 3, "patch_sizes must be a 3-element tuple (d, h, w)"
    p_d, p_h, p_w = patch_sizes
    
    # 验证输入和标签张量形状匹配
    B, C, D, H, W = input_tensor.shape
    assert C == 1, "input_tensor must have 1 channel dimension"
    assert label_tensor.shape == (B, 1, D, H, W), "label_tensor shape must match input_tensor"
    assert isinstance(mask_ratios, dict), "mask_ratios must be a non-empty dictionary"
    
    # 对原始D, H, W进行填充，确保能被patch大小整除
    new_D = int(p_d * np.ceil(D / p_d))
    new_H = int(p_h * np.ceil(H / p_h))
    new_W = int(p_w * np.ceil(W / p_w))
    assert new_D >= D and new_H >= H and new_W >= W, "new dimensions must be larger than original"
    # 初始化掩码列表（每个样本单独处理）
    masks = []
    
    for b in range(B):
        # 对单个样本的输入和标签进行填充
        input_single = input_tensor[b:b+1]
        label_single = label_tensor[b:b+1]
        
        # 填充输入和标签到新尺寸
        pad_d = new_D - D
        pad_h = new_H - H
        pad_w = new_W - W
        pad = (0, pad_w, 0, pad_h, 0, pad_d)  # (左,右,上,下,前,后)
        
        padded_label = torch.nn.functional.pad(label_single, pad, mode='constant', value=0)
        padded_label = padded_label.int()  # 确保标签为整数类型
        
        # 初始化全1掩码（1表示需要mask的区域）
        mask = torch.ones(1, 1, new_D, new_H, new_W, device=input_tensor.device)
        
        # 计算各维度上的patch数量
        d = new_D // p_d
        h = new_H // p_h
        w = new_W // p_w
        #total_patches = d * h * w
        
        # 将标签重塑为patch结构以确定每个patch的主要标签
        # 重塑为(1, 1, d_patch, p_d, h_patch, p_h, w_patch, p_w)
        label_patched = padded_label.reshape(1, 1, d, p_d, h, p_h, w, p_w)
        # 合并patch内部维度以统计主要标签
        label_patched = label_patched.permute(0,1,2,4,6,3,5,7)
        label_patched = label_patched.reshape(1, d, h, w, p_d * p_h * p_w)
        
        # 确定每个patch的主要标签（出现次数最多的标签）
        label_m = label_patched.clone()
        label_m = label_m.reshape(1*d*h*w, -1)
        label_m = torch.sum(label_m, dim=1) > (p_d * p_h *p_w) * threshold

        p_tr_num = len(label_m[label_m == 1])
        p_tr_indxs = label_m.nonzero().squeeze()
        p_fl_num = len(label_m[label_m == 0])
        p_fl_indxs = (label_m == 0).nonzero().squeeze()

        #mask = mask.reshape(1, 1, new_D, new_H, new_W)
        mask = mask.reshape(1, 1, d, p_d, h, p_h, w, p_w)
        mask = mask.permute(0,1,2,4,6,3,5,7)
        mask = mask.reshape(1, d, h, w, p_d * p_h * p_w)
        mask = mask.reshape(1*d*h*w, -1)

        tr_keep_num = p_tr_num * (1 - mask_ratios[1]) #前景
        fl_keep_num = p_fl_num * (1 - mask_ratios[0]) #背景
        p_tr_indxs_select = random.sample(p_tr_indxs.tolist(), int(tr_keep_num))
        p_fl_indxs_select = random.sample(p_fl_indxs.tolist(), int(fl_keep_num))

        zero = torch.zeros(size=(p_d * p_h * p_w,))
        mask[p_tr_indxs_select] = zero
        mask[p_fl_indxs_select] = zero

        mask = mask.reshape(-1, p_d, p_h, p_w)
        mask = mask.reshape(1, d, h, w, p_d, p_h, p_w)
        mask = mask.permute(0, 1, 4, 2, 5, 3, 6) #* d, h, w, p_d, p_h, p_w -> d, p_d, h, p_h, w, p_w
        mask = mask.reshape(1, new_D, new_H, new_W)
        mask = mask.unsqueeze(0)

        # patch_labels = []
        # for i in range(d):
        #     for j in range(h):
        #         for k in range(w):
        #             # 获取当前patch的所有标签
        #             patch_vals = label_patched[0, i, j, k]
        #             # 统计标签出现次数
        #             vals, counts = torch.unique(patch_vals, return_counts=True)
        #             # 找到出现次数最多的标签
        #             dominant_label = vals[torch.argmax(counts)].item()
        #             patch_labels.append((i, j, k, dominant_label))
        
        # # 按标签分组patch索引
        # label_to_patches = {}
        # for label in mask_ratios.keys():
        #     label_to_patches[label] = []
        
        # for i, j, k, label in patch_labels:
        #     if label in label_to_patches:
        #         label_to_patches[label].append((i, j, k))
        #     else:
        #         # 对于不在mask_ratios中的标签，默认不mask
        #         label_to_patches.setdefault(label, []).append((i, j, k))
        
        # # 重塑掩码为patch结构以便处理
        # mask = mask.reshape(1, 1, d, p_d, h, p_h, w, p_w)
        
        # # 对每个标签类别应用相应的mask比例
        # for label, ratio in mask_ratios.items():
        #     patches = label_to_patches.get(label, [])
        #     if not patches:
        #         continue
                
        #     # 计算需要保留的patch数量
        #     num_patches = len(patches)
        #     len_keep = int(num_patches * (1 - ratio))
        #     len_keep = max(0, min(len_keep, num_patches))  # 确保在有效范围内
            
        #     # 随机选择要保留的patch
        #     keep_indices = torch.randperm(num_patches, device=input_tensor.device)[:len_keep]
        #     keep_patches = [patches[idx] for idx in keep_indices]
            
        #     # 将保留的patch设为0（不mask）
        #     for i, j, k in keep_patches:
        #         mask[0, 0, i, :, j, :, k, :] = 0
        
        # # 将掩码重塑回空间尺寸
        # mask = mask.reshape(1, 1, new_D, new_H, new_W)
        if_random_affine = False
        #! 可选：应用随机仿射变换增强掩码多样性. RandAffine,默认填充为0,不能用
        if if_random_affine:
            affine_transform = RandAffine(
                rotate_range=[(-np.pi/6, np.pi/6)] * 3,  # 三个轴的旋转范围（±30度）
                prob=0.8,
                mode="nearest",
                padding_mode="constant"
            )
            # 应用变换并四舍五入确保掩码值为0或1
            result = affine_transform(mask[0].cpu())  # 暂时移到CPU处理
            mask[0] = torch.round(result).to(input_tensor.device)
        
        # 裁剪回原始输入尺寸
        new_mask = RandSpatialCrop(
            roi_size=(D, H, W), 
            random_size=False
        )(mask[0]).unsqueeze(0)
        
        masks.append(new_mask)
    
    # 合并所有样本的掩码
    mask = torch.cat(masks, dim=0)
    
    return mask

# 使用示例
if __name__ == "__main__":
    label_path = '/root/shared-nvme/lyx/Project/Nerve/Data/Sert-Stanford/train/labels-8bit/annotation-1.tif'
    save_dir = '/root/shared-nvme/lyx/Project/Nerve/Trial/Res/trail'
    save_path = os.path.join(save_dir, os.path.basename(label_path))
    label = sitk.ReadImage(label_path)
    label = sitk.GetArrayFromImage(label)

    label = torch.from_numpy(label)
    label = label.unsqueeze(0).unsqueeze(0)

    mask_ratios = {0: 0.2, 1: 0.25}

    mask = mask_generator3D_v2(
        input_tensor=label,
        label_tensor=label,
        mask_ratios=mask_ratios,
        patch_sizes=(3,3,3),
        if_random_affine=False,
        threshold=0.2
    )

    label = label * torch.abs(1-mask)
    label = torch.squeeze(label).numpy()

    label = label.astype(np.uint8)

    tiff.imsave(
        save_path,
        label,
        photometric='minisblack',
    )


    # # 创建示例输入和标签张量
    # B, C, D, H, W = 2, 1, 10, 10, 10
    # input_tensor = torch.randn(B, C, D, H, W)
    # # 创建标签张量（0表示背景，1表示前景）
    # label_tensor = torch.randint(0, 2, (B, C, D, H, W), dtype=torch.uint8)
    
    # # 定义不同标签的mask比例
    # mask_ratios = {0: 0.2, 1: 0.5}  # 背景mask 20%，前景mask 50%
    # patch_sizes = (2, 3, 2)
    
    # # 生成掩码
    # mask = mask_generator3D_v2(
    #     input_tensor=input_tensor,
    #     label_tensor=label_tensor,
    #     mask_ratios=mask_ratios,
    #     patch_sizes=patch_sizes,
    #     if_random_affine=False
    # )
    
    # print(f"生成的掩码形状: {mask.shape}")
    # print(f"掩码值范围: [{mask.min()}, {mask.max()}]")
    
    # # 验证不同区域的mask比例
    # for b in range(B):
    #     for label in [0, 1]:
    #         label_mask = (label_tensor[b] == label).float()
    #         if label_mask.sum() == 0:
    #             continue
    #         masked_area = (mask[b] * label_mask).sum() / label_mask.sum()
    #         print(f"样本 {b} 中标签 {label} 的实际mask比例: {masked_area:.4f}")
