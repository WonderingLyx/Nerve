import os
import numpy as np
from os import path as osp
import torch
# import cv2
from torch.utils.data import Dataset
from PIL import Image
from monai.transforms import *
from monai.data import CacheDataset
import pytorch_lightning as pl
import os
import numpy as np
from os import path as osp
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset
from PIL import Image
from monai.transforms import *
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch





class NerveDataset_MIM():
    def __init__(self, root_dir,is_train=True, is_test=False, is_MakePatches=True, transform_id=0):
        self.is_train = is_train
        self.is_test = is_test
        self.is_MakePatches = is_MakePatches
        self.transform_id = transform_id
        
        self.root_dir = root_dir
        self.train_dir = osp.join(self.root_dir, 'training')
        self.test_dir = osp.join(self.root_dir, 'test')
        
        self.data_dir = self.train_dir if is_train else self.test_dir
        for r, dirs, images in os.walk(osp.join(self.data_dir, 'images')):
            self.images = map(lambda x: osp.join(r, x), filter(
                lambda x: x.endswith('tif'), images))
            self.images = sorted(list(self.images))
            break

        for r, dirs, images in os.walk(osp.join(self.data_dir, 'labels')):
            self.seg = map(lambda x: osp.join(r, x), filter(
                lambda x: x.endswith('tif'), images))
            self.seg = sorted(list(self.seg))
            break
        
        self.data_dict = [{"image": image_path, "label": label_path} for
                          (image_path, label_path) in zip(self.images, self.seg)]
        self.init_transforms()

    def init_transforms(self):
        if self.transform_id == 0:
            self.train_transforms = Compose([
                LoadImaged(keys=["image", "label"],
                           reader="PILReader"),
                EnsureChannelFirstd(keys=["image", "label"]),

                ScaleIntensityRanged(keys=["image", "label"], a_min=0, a_max=255,
                                     b_min=0, b_max=1, clip=True),
                RandCropByPosNegLabeld(
                    keys=["image", "label"], label_key="label", spatial_size=[128, 128], pos=3, neg=1, num_samples=4
                ) if self.is_MakePatches else None,
                RandFlipd(keys=["image", "label"], prob=0.5),
                RandRotate90d(keys=["image", "label"], prob=0.5),
                AsDiscreted(keys=["label"], rounding="torchrounding"),
                EnsureTyped(keys=["image", "label"])]
                                    
            )

            self.test_transforms = Compose([
                LoadImaged(keys=["image",  "label"],
                           reader="PILReader"),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(keys=["image", "label"], a_min=0, a_max=255,
                                     b_min=0, b_max=1, clip=True),
                RandCropByPosNegLabeld(
                    keys=["image", "label"], label_key="label", spatial_size=[128, 128], pos=3, neg=1, num_samples=4
                ) if self.is_MakePatches else None,
                AsDiscreted(keys=["label"], rounding="torchrounding"),
                EnsureTyped(keys=["image", "label"])]
            )


    def return_TrainDataset(self):
        assert self.is_train
        assert not self.is_test
        ds = CacheDataset(
            self.data_dict, transform=self.train_transforms, cache_rate=1)
        return ds

    def return_TestDataset(self):
        assert self.is_test
        assert not self.is_train
        ds = CacheDataset(
            self.data_dict, transform=self.test_transforms, cache_rate=1)
        return ds






class NerveDataset_MIM_PL(pl.LightningDataModule):
    def __init__(self, Config, data_dir='./data/DRIVE'):
        super(NerveDataset_MIM_PL, self).__init__()
        self.config = Config
        self.data_dir = data_dir
        pass

    def train_dataloader(self):
        train_ds = NerveDataset_MIM(root_dir=self.data_dir,is_train=True, is_test=False).return_TrainDataset()
        train_dataloader = DataLoader(train_ds, batch_size=self.config["batch_size"], shuffle=True,num_workers=2)
        return train_dataloader

    def val_dataloader(self):
        val_ds = NerveDataset_MIM(root_dir=self.data_dir,is_train=False, is_test=True).return_TestDataset()
        val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False,num_workers=1)
        return val_dataloader

    def test_dataloader(self):
        test_ds = NerveDataset_MIM(root_dir=self.data_dir,is_train=False, is_test=True).return_TestDataset()
        test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False,num_workers=1)
        return test_dataloader
    


# main
if __name__ == "__main__":
    config = {
        "batch_size": 2,
    }
    data_dir = "//root/shared-nvme/lyx/Project/Nerve/Data/Nerve"
    PL_dataset = NerveDataset_MIM_PL(config, data_dir=data_dir)
    train_loader = PL_dataset.train_dataloader()
    
    for data in train_loader:
        print(data)
    
    
    