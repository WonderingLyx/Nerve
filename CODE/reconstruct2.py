import numpy as np
from PIL import Image

def load_tiff_stack(file_path):
    """加载一个3D TIFF文件，返回一个NumPy数组"""
    images = []
    with Image.open(file_path) as img:
        for i in range(img.n_frames):
            img.seek(i)
            images.append(np.array(img))
    return np.stack(images, axis=0)  # 堆叠形成3D数组

def save_tiff_stack(images, output_path):
    """将一个NumPy数组保存为3D TIFF文件"""
    first_image = Image.fromarray(images[0])
    first_image.save(output_path, save_all=True, append_images=[Image.fromarray(img) for img in images[1:]])

def reconstruct_original_from_coronal(stacked_3d_image, original_shape):
    """
    从冠状面堆叠的2D切片恢复原始3D图像。
    
    Parameters:
        stacked_3d_image (np.ndarray): 堆叠的3D图像 (depth, height, width)。
        original_shape (tuple): 原始3D图像的形状 (depth, height, width)。
    """
    # 分解形状
    depth, height, width = original_shape

    # 检查堆叠维度是否匹配
    if stacked_3d_image.shape[0] != width:
        raise ValueError("堆叠的图像深度���原始宽度不匹配，检查输入数据或原始形状！")

    # 重构：从堆叠的深度恢复为原始深度
    reconstructed = np.zeros(original_shape, dtype=stacked_3d_image.dtype)
    for i in range(width):
        reconstructed[:, :, i] = stacked_3d_image[i]

    return reconstructed

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

# 将示例代码放在 if __name__ == "__main__": 下
if __name__ == "__main__":
    # 示例代码
    stacked_file = "/mnt/40B2A1DBB2A1D5A6/fxj/output_train/3D_predictions_cldice+150T+3Dmse_5/Image_0_Coronal_epoch_0_view_coronal_label.tiff"
    original_shape = (150, 150, 150)
    stacked_image = load_tiff_stack(stacked_file)
    reconstructed_image = reconstruct_original_from_coronal(stacked_image, original_shape)
    final_image = apply_transformations(reconstructed_image)
    output_file = "/mnt/40B2A1DBB2A1D5A6/fxj/SAM2-UNet3/data/test/reconstructed_3d_image.tif"
    save_tiff_stack(final_image, output_file)

