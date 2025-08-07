import os
import numpy as np
from PIL import Image
import tifffile as tiff  # 用于加载 3D tiff 图像
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import SimpleITK as sitk

class ToTensor(object):
    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {
            'image': F.resize(image, self.size),
            'label': F.resize(label, self.size, interpolation=InterpolationMode.NEAREST)
        }

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):  # 修改为 RGB 图像的均值和标准差
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}


class ImageDataset(Dataset):
    def __init__(self,image_root, gt_root, size, mode,  subset_start=None, 
                 subset_size=None, is_random_flip=False, flip_rate=3, **kwargs):
        
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root)]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root)]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        if len(self.images) != len(self.gts):
            raise ValueError(f"Number of images ({len(self.images)}) and labels ({len(self.gts)}) are not the same!")
        
        volumns_num = len(self.images) // 150
        self.images = self.images[:volumns_num * 150]
        self.gts = self.gts[:volumns_num * 150]

        if subset_start is not None and subset_size is not None:
            assert (subset_start + subset_size) < len(self.images)
            self.images = self.images[subset_start:subset_start+subset_size]
            self.gts = self.gts[subset_start: subset_start+subset_size]

        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])
        

    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])

        #* fake img_index
        img_index = idx // 150
        data = {'image': image, 'label': label}
        data = self.transform(data)

        data['image_index'] = img_index
        return data
    
    def get_original_3d(self, image_index):
        if image_index < 0 or (image_index >= len(self.images) // 150):
            raise IndexError(f"Image index {image_index} out of range.")
        
        img_idxes = range(image_index*150, (image_index+1)*150)
        volumns = [np.array(self.rgb_loader(self.images[i]))[:,:,0][np.newaxis,:,:] for i in img_idxes]

        return np.array(volumns)
    
    def get_original_label_3d(self, image_index):
        if image_index < 0 or (image_index >= len(self.images) // 150):
            raise IndexError(f"Image index {image_index} out of range.")
        
        img_idxes = range(image_index*150, (image_index+1)*150)
        volumns = [np.array(self.binary_loader(self.gts[i])) for i in img_idxes]

        return np.array(volumns)
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        
class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode, view='axial', subset_start=None, subset_size=None, **kwargs):
        """
        image_root: 3D 图像的目录（tiff 格式）
        gt_root: 3D 标签的目录（tiff 格式）
        size: 输出 2D 图像的尺寸
        mode: 'train' 或 'val'
        view: 切片视角，支持 'axial'、'coronal' 和 'sagittal'
        subset_start: 子集开始索引
        subset_size: 子集大小
        """
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.tiff') or f.endswith('.tif')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.tiff') or f.endswith('.tif')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        # 修改子集选择逻辑
        if subset_start is not None and subset_size is not None:
            end_idx = min(subset_start + subset_size, len(self.images))
            self.images = self.images[subset_start:end_idx]
            self.gts = self.gts[subset_start:end_idx]
            print(f"Loading subset from {subset_start} to {end_idx}, total {len(self.images)} images")

        if len(self.images) != len(self.gts):
            raise ValueError(f"Number of images ({len(self.images)}) and labels ({len(self.gts)}) are not the same!")

        self.size = size
        self.mode = mode
        self.view = view

        # 定义数据增强和标准化
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __len__(self):
        # 修改返回长度计算
        num_images = len(self.images)
        slices_per_image = 150  # 每个3D图像的切片数
        total_slices = num_images * slices_per_image
        return total_slices

    def load_tiff(self, path):
        """加载 3D tiff 图像并返回为一个 numpy 数组"""
        with tiff.TiffFile(path) as tif:
            return tif.asarray()

    def __getitem__(self, idx):
        # 添加数据验证
        file_idx = idx // 150
        slice_idx = idx % 150

        # 加载并验证数据
        try:
            image_3d = self.load_tiff(self.images[file_idx])
            label_3d = self.load_tiff(self.gts[file_idx])
            
            # 验证数据维度
            if image_3d.ndim != 3 or label_3d.ndim != 3:
                raise ValueError(f"Invalid dimensions: image {image_3d.shape}, label {label_3d.shape}")

            # 根据视角获取切片
            if self.view == 'axial':
                image_slice = image_3d[slice_idx, :, :]
                label_slice = label_3d[slice_idx, :, :]
            elif self.view == 'coronal':
                image_slice = image_3d[:, slice_idx, :]
                label_slice = label_3d[:, slice_idx, :]
            else:
                raise ValueError(f"Invalid view: {self.view}")

            # 确保数据类型和范围正确
            image_slice = image_slice.astype(np.uint8)
            label_slice = label_slice.astype(np.uint8)

            # 转换为PIL图像
            image_slice = Image.fromarray(image_slice).convert("RGB")
            label_slice = Image.fromarray(label_slice).convert("L")

            # 应用变换
            data = {'image': image_slice, 'label': label_slice}
            data = self.transform(data)

            # 添加额外信息
            data['name'] = os.path.basename(self.images[file_idx])
            data['image_index'] = file_idx

            return data

        except Exception as e:
            print(f"Error loading data at index {idx}: {str(e)}")
            # 返回一个有效的替代数据
            return self.__getitem__((idx + 1) % self.__len__())

    def get_original_3d(self, image_index):
        """
        根据索引返回完整的原始 3D 图像（未裁剪、未处理）。
        """
        if image_index < 0 or image_index >= len(self.images):
            raise IndexError(f"Image index {image_index} out of range.")
        return self.load_tiff(self.images[image_index])

    def get_original_label_3d(self, image_index):
        """
        根据索引返回完整的原始 3D 标签（未裁剪、未处理）。
        """
        if image_index < 0 or image_index >= len(self.gts):
            raise IndexError(f"Label index {image_index} out of range.")
        return self.load_tiff(self.gts[image_index])
    def get_file_name(self, image_index):
        if image_index < 0 or image_index >= len(self.gts):
            raise IndexError(f"Label index {image_index} out of range.")

        return os.path.basename(self.images[image_index])



class TestDataset(Dataset):
    def __init__(self, image_root, size=352, view='axial'):
        """
        加载 3D 测试图像数据，并按指定视角切片。
        :param image_root: 3D 图像文件路径 (可以是目录或者单个 tiff 文件)。
        :param size: 切片的目标尺寸。
        :param view: 切片视角，支持 'axial'、'coronal' 和 'sagittal'。
        """
        if os.path.isdir(image_root):
            self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.tiff') or f.endswith('.tif')]
        elif os.path.isfile(image_root) and (image_root.endswith('.tiff') or image_root.endswith('.tif')):
            self.images = [image_root]
        else:
            raise ValueError(f"[ERROR] Invalid image_root: {image_root}. It should be a directory or a .tiff file.")

        if not self.images:
            raise ValueError(f"[ERROR] No valid .tiff or .tif files found in {image_root}.")

        self.images = sorted(self.images)
        print(f"[INFO] Loaded {len(self.images)} 3D images from {image_root}")

        self.size = size
        self.view = view

        # 定义数据变换，与训练时一致
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_tiff(self, path):
        """加载 3D tiff 图像并返回为一个 numpy 数组"""
        with tiff.TiffFile(path) as tif:
            return tif.asarray()

    def __getitem__(self, idx):
        """
        返回完整的 3D 图像的所有切片。
        :param idx: 索引
        :return: 切片张量列表、原始 3D 图像形状、文件名、原始切片列表
        """
        image_path = self.images[idx]
        image_3d = self.load_tiff(image_path)

        print(f"[DEBUG] Loaded 3D image shape: {image_3d.shape}")

        # 根据指定视角进行切片
        slices = []
        original_slices = []  # 保存未处理的原始切片

        if self.view == 'axial':  # XY 平面
            for i in range(image_3d.shape[0]):
                slice_2d = image_3d[i, :, :]  # 提取第 i 个切片
                original_slices.append(slice_2d)  # 未处理的原始切片

                slice_2d = Image.fromarray(slice_2d.astype(np.uint8))  # 转为 PIL 图像
                slice_2d = slice_2d.convert("RGB")  # 转为伪 RGB 图像
                slice_2d = self.transform(slice_2d)  # 应用预处理
                slices.append(slice_2d)

        elif self.view == 'coronal':  # XZ 平面
            for i in range(image_3d.shape[1]):
                slice_2d = image_3d[:, i, :]  # 提取第 i 个切片
                original_slices.append(slice_2d)  # 未处理的原始切片

                slice_2d = Image.fromarray(slice_2d.astype(np.uint8))
                slice_2d = slice_2d.convert("RGB")
                slice_2d = self.transform(slice_2d)
                slices.append(slice_2d)

        elif self.view == 'sagittal':  # YZ 平面
            for i in range(image_3d.shape[2]):
                slice_2d = image_3d[:, :, i]  # 提取第 i 个切片
                original_slices.append(slice_2d)  # 未处理的原始切片

                slice_2d = Image.fromarray(slice_2d.astype(np.uint8))
                slice_2d = slice_2d.convert("RGB")
                slice_2d = self.transform(slice_2d)
                slices.append(slice_2d)

        else:
            raise ValueError(f"Invalid view specified: {self.view}. Choose from 'axial', 'coronal', 'sagittal'.")

        # 返回所有切片张量、原始 3D 图像的形状、文件名、原始切片列表
        return torch.stack(slices), image_3d.shape, os.path.basename(image_path), original_slices

    def __len__(self):
        """
        返回数据集中 3D 图像的数量。
        """
        return len(self.images)