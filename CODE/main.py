import argparse
from train import main
from test import test

parser = argparse.ArgumentParser("SAM2-UNet")
# paths to model or data or results
parser.add_argument(
    "--hiera_path",
    default='sam2_hiera_l',
    help="SAM2 pretrained Hiera yaml"
)

parser.add_argument(
    "--pretrain_path",
    default="/root/shared-nvme/Project/Nerve/Pretrain/sam2_hiera_large.pt",
    help="Path to the SAM2 pretrained model weights",
)

parser.add_argument(
    "--train_image_path",
    default="/root/shared-nvme/Project/Nerve/Data/Sert-Stanford/train/volumes-8bit",
    help="Path to the 3D images"
)

parser.add_argument(
    "--train_mask_path",
    default="/root/shared-nvme/Project/Nerve/Data/Sert-Stanford/train/labels-8bit",
    help="Path to the 3D masks"
)

parser.add_argument(
    "--save_path",
    default="/root/shared-nvme/Project/Nerve/Trial/RES",
    help="Path to store the training results log, weights"
)

parser.add_argument(
    "--output_path",
    default="",
    help="Path to store 3D prediction images"
)

#! experiment set
parser.add_argument("--test_id", type=str, default='sert-dsc-0', help='experiment name')


#! model set
parser.add_argument("--input_channel", type=int, default=3, help="input data channels")
parser.add_argument("--dsc_number", type=int, default=16, help="dsc conv num")
parser.add_argument("--adadim", type=int, default=32, help="adapter dim default to 32")
parser.add_argument("--rfbdim", type=int, default=256, help="RFB module dim default to 256")

# training set
parser.add_argument("--epoch", type=int, default=10, help="Training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--batch_size", default=5, type=int)  # 批量大小
parser.add_argument("--device", type=str, default='cuda:0', help="GPU id")
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--lambda_3d", type=float, default=0.1, help="Weight for 3D MSE loss (lambda). Default is 0.1.")

#! test set
parser.add_argument(
    "--device_test",
    default="cuda:0",
    help="GPU id for test"
)
parser.add_argument(
    "--checkpoint",
    default="/root/shared-nvme/Project/Nerve/Trial/RES/Model/sert-dsc-0-epoch-10.pth",
    help="Path to the model checkpoint"
)
parser.add_argument(
    "--test_image_path",
    default="/root/shared-nvme/Project/Nerve/Data/Sert-Stanford/test/test/volume-8bit",
    help="Path to the 3D test images"
)
parser.add_argument(
    "--predict_path",
    default="/root/shared-nvme/Project/Nerve/Trial/RES/Predicts",
    help="Path to save 3D prediction results and original slices"
)

args = parser.parse_args()

# 训练
#main(args)

# 测试
test(args)
