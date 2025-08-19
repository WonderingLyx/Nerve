from DeepClosing3D import DeepClosing
import yaml
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import tifffile as tiff

check_point_path = '/root/shared-nvme/lyx/Project/Nerve/Trial/Res/Model/DC/epoch=49-step=800.ckpt'
config_path = '/root/shared-nvme/lyx/Project/Nerve/Trial/Nerve-sam2post/CONFIG/Nerve.yaml'
seg_path = '/root/shared-nvme/lyx/Project/Nerve/Trial/Res/Predicts/Predicts_3D'
seg_DC_path = '/root/shared-nvme/lyx/Project/Nerve/Trial/Res/Predicts/Predicts_DC'

f = open(config_path, 'r', encoding='utf-8')
cont = f.read()
config = yaml.load(cont, Loader=yaml.FullLoader)

model = DeepClosing.load_from_checkpoint(
    checkpoint_path=check_point_path,
    config=config
)

img_paths = [ f for f in os.listdir(seg_path) if f.endswith('.tiff')]

import os

def get_sort_key(filename):
    """提取文件名中的编号"""
    # 提取volume后的数字X（如从"volume-2.tiff_prediction_0000"中提取2）
    x_part = filename.split('.')[0].split('-')[-1]
    x = int(x_part)
    
    # 提取prediction后的数字Y（如从"volume-2.tiff_prediction_0000"中提取0000→0）
    # y_part = filename.split('.')[1].split('_')[-1]  # 忽略可能的后缀（如.tif）
    # y = int(y_part)
    
    return x

# 按规则排序
sorted_files = sorted(img_paths, key=get_sort_key)

#index = get_sort_key(sorted_files[0])[0]
for pred in tqdm(sorted_files):
    idx = get_sort_key(pred)
    
    pred_path = os.path.join(seg_path, pred)

    res = model.DeepClosing(pred_path)
    
    dc = res['T_dc']
    dc = dc.cpu().numpy()
    base_name = pred.split('.')[0]
    
    # num = pred.split('.')[1].split('_')[-1]
    # base_name = base_name + '-' + num

    dc = (dc * 255).astype(np.uint8)
            
    # 3. 保存为TIFF和GIF
    tif_save_path = os.path.join(seg_DC_path, f"{base_name}-DC.tif")
    #gif_save_path = os.path.join(seg_DC_path, f"{base_name}-DC.gif")
    
    # Image.fromarray(dc).save(tif_save_path)
    # Image.fromarray(dc).save(gif_save_path, format='GIF')

    tiff.imwrite(
        tif_save_path,
        dc,
        photometric='minisblack'
    )

    
    print(f"Finish process {base_name}")
    #index = idx

        
