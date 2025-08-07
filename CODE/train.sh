python train.py  \
--hiera_path "sam2_hiera_l" \
--pretrain_path "/mnt/40B2A1DBB2A1D5A6/lyx/project/UPLOAD/Brain/MODEL/Sam2/sam2_hiera_large.pt" \
--train_image_path "/mnt/40B2A1DBB2A1D5A6/lyx/project/MedSam2/SAM-Unet/DATA/P28/train/8bit/volumes" \
--train_mask_path "/mnt/40B2A1DBB2A1D5A6/lyx/project/MedSam2/SAM-Unet/DATA/P28/train/8bit/labels" \
--save_path "/mnt/40B2A1DBB2A1D5A6/lyx/project/Nerve/nerve/Trial/RESULTS" \
--device "cuda:1" \
--epoch 1 \
--lr 0.001 \
--batch_size 1
