import os
from PIL import Image
import numpy as np
import tifffile as tiff

def split_3d_tiff(input_path, output_dir=None, prefix="slice_"):
    """
    将三维TIFF文件分割为多个二维TIFF图像
    
    参数:
        input_path: 三维TIFF文件的路径
        output_dir: 输出二维TIFF的目录，默认为输入文件所在目录
        prefix: 输出文件的前缀，默认为"slice_"
    """
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    
    try:
        # 打开三维TIFF文件
        with Image.open(input_path) as img:
            slice_idx = 0
            while True:
                try:
                    # 选择当前页（切片）
                    img.seek(slice_idx)
                    
                    # 生成输出文件名
                    input_filename = os.path.splitext(os.path.basename(input_path))[0]
                    output_filename = f"{prefix}{input_filename}_{slice_idx:04d}.tif"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # 保存当前切片为二维TIFF
                    img.save(output_path)
                    print(f"已保存: {output_path}")
                    
                    slice_idx += 1
                except EOFError:
                    # 所有切片处理完毕
                    break
        print(f"转换完成，共生成 {slice_idx} 个二维TIFF文件")
    except Exception as e:
        print(f"转换失败: {str(e)}")

def get_sort_key(filename):
    """提取文件名中的X和Y，返回排序键(X, Y)"""
    # 提取volume后的数字X（如从"volume-2.tiff_prediction_0000"中提取2）
    x_part = filename.split('-')[1]
    x = int(x_part)
    
    # 提取prediction后的数字Y（如从"volume-2.tiff_prediction_0000"中提取0000→0）
    y_part = filename.split('-')[2] # 忽略可能的后缀（如.tif）
    y = int(y_part)
    
    return (x,y)

def merge_to_3d_tiff(input_dir, output_path=None, prefix="slice_", sort_key=None):
    """
    将多个二维TIFF图像合并为三维TIFF文件
    
    参数:
        input_dir: 包含二维TIFF文件的目录
        output_path: 输出三维TIFF的路径，默认为输入目录下的3d_merged.tif
        prefix: 用于筛选文件的前缀，默认为"slice_"
        sort_key: 自定义排序函数，用于确定切片顺序，默认为按文件名中的数字排序
    """
    # 确定输出路径
    if output_path is None:
        output_path = os.path.join(input_dir, "3d_merged.tif")
    
    # 获取所有符合条件的TIFF文件
    tif_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.tif')) and f.startswith(prefix)
    ]
    
    if not tif_files:
        print("未找到符合条件的TIFF文件")
        return
    
    # 定义默认排序函数（按文件名中的数字排序）
    def default_sort_key(filename):
        # 提取文件名中的数字部分
        import re
        match = re.search(r'_(\d+)\.', filename)
        if match:
            return int(match.group(1))
        return 0
    
    # 使用自定义排序函数或默认函数
    if sort_key is None:
        sort_key = default_sort_key
    
    # 对文件进行排序
    tif_files.sort(key=sort_key)
    #tif_paths = [os.path.join(input_dir, f) for f in tif_files]
    
    # 打开第一个图像作为基础
    images = []
    x_idx, y_idx = sort_key(tif_files[0])
    for idx, f in enumerate(tif_files):
        x, y = sort_key(f)
        path = os.path.join(input_dir, f)
        target_path = os.path.join(output_path, f"volume-{x_idx}-DC.tif")

        if x != x_idx:
            assert y == 0
            images = np.array(images, dtype=np.uint8)
            tiff.imwrite(
                target_path,
                images,
                photometric='minisblack'  # 灰度图像模式
                #metadata={'axes': 'ZYX'}   # 明确指定轴顺序，便于后续读取
            )
            images = []
            print(f"Finish volume-{x_idx}")
            x_idx = x

        img = Image.open(path)
        img = np.array(img)
        assert img.shape == (150,150)
        images.append(img.tolist())

        if idx == len(tif_files) - 1:
            if y == 149:
                tiff.imwrite(
                    os.path.join(output_path, f'volume-{x}-DC.tiff'),
                    images,
                    photometric='minisblack',  # 灰度图像模式
                    metadata={'axes': 'ZYX'}   # 明确指定轴顺序，便于后续读取
                )
                print(f"Finish volume-{x}")
                


# 示例用法
if __name__ == "__main__":
    # 三维TIFF文件路径
    # input_tiff_dir = "/root/shared-nvme/lyx/Project/Nerve/Trial/Res/Predicts/Predicts_3D"  # 替换为你的三维TIFF文件路径
    # output_directory = "/root/shared-nvme/lyx/Project/Nerve/Trial/Res/Predicts/Predicts"
    # for f in os.listdir(input_tiff_dir):
    #     input_tiff = os.path.join(input_tiff_dir, f)  
    #     # 执行转换
    #     split_3d_tiff(
    #         input_path=input_tiff,
    #         output_dir=output_directory,
    #         prefix=""  # 输出文件前缀
    #     )


    input_tiff_dir = '/root/shared-nvme/lyx/Project/Nerve/Trial/Res/Predicts/Predicts_DC'
    output_directory = '/root/shared-nvme/lyx/Project/Nerve/Trial/Res/Predicts/Predicts_3D_DC/'

    merge_to_3d_tiff(input_tiff_dir, output_directory, prefix='', sort_key=get_sort_key)
    