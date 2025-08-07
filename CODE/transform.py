import os
import numpy as np
from PIL import Image
import tifffile as tiff
import SimpleITK as sitk

def volumes_16_to_8bit(img_dir, output_dir):

    assert os.path.exists(img_dir)
    assert os.path.exists(output_dir)

    imgs = sorted(os.listdir(img_dir))

    for im in imgs:
        print(f"Processing {im}")
        img_path = os.path.join(img_dir, im)
        
        img = sitk.ReadImage(img_path)

        img_8bit = sitk.RescaleIntensity(img, outputMinimum=0, outputMaximum=255)
        img_8bit = sitk.GetArrayFromImage(img_8bit)

        
        img_8bit = img_8bit.astype(np.uint8)
        
        new_path = os.path.join(output_dir, im)

        with tiff.TiffWriter(new_path) as tif:
            tif.write(
                img_8bit
            )
        


if __name__ == "__main__":

    source = "/root/shared-nvme/Project/Nerve/Data/Sert-Stanford/train/labels"
    target = "/root/shared-nvme/Project/Nerve/Data/Sert-Stanford/train/labels-8bit"

    volumes_16_to_8bit(source, target)
