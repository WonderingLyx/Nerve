import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
#from S3_DSConv_pro import DSConv_pro as DSConv
from omegaconf import OmegaConf

import numpy as np
from Simple_Point_Erosion_Module import Simple_Point_Erosion_module
import argparse
import SimpleITK as sitk

backbone_channel_list = {
    "sam2_hiera_l" : [1152, 576, 288, 144],
    "sam2_hiera_s" : [768, 384, 192, 96]
}


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Adapter(nn.Module):
    def __init__(self, blk, args) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, args.adadim),
            nn.GELU(),
            nn.Linear(args.adadim, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x
    
class DSCBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        if_offset: bool = True,
        device: str = "cuda:0"
    ):
        super(DSCBlock, self).__init__()

        self.conv0 = EncoderConv(input_channels, output_channels)
        self.convx = DSConv(
            input_channels,
            output_channels,
            kernel_size,
            extend_scope,
            0,
            if_offset,
            device,
        )
        self.convy = DSConv(
            input_channels,
            output_channels,
            kernel_size,
            extend_scope,
            1,
            if_offset,
            device,
        )
        self.conv1 = EncoderConv(3 * output_channels, output_channels)

    def forward(self, x):
        x_0 = self.conv0(x)
        x_x = self.convx(x)
        x_y = self.convy(x)
        x_1 = self.conv1(torch.cat([x_0, x_x, x_y], dim=1))

        return x_1



class SAM2UNet(nn.Module):
    def __init__(self, model_cfg=None, checkpoint_path=None, args=None, device=None) -> None:
        super(SAM2UNet, self).__init__()  

        #* sam2 
        if model_cfg is None: 
            model_cfg = "sam2_hiera_l.yaml"
        else:
            sam_types = ['sam2_hiera_l', 'sam2_hiera_s']
            assert model_cfg in sam_types
            model_cfg = model_cfg + '.yaml'

        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path, device=device)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []

        #* 加入Adapter
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block, args)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        self.backbone_channel_list = backbone_channel_list[model_cfg.split('.')[0]]
        ch_list = self.backbone_channel_list
        #* 统一channel数量
        self.rfb1 = RFB_modified(ch_list[3], args.rfbdim)
        self.rfb2 = RFB_modified(ch_list[2], args.rfbdim)
        self.rfb3 = RFB_modified(ch_list[1], args.rfbdim)
        self.rfb4 = RFB_modified(ch_list[0], args.rfbdim)
        self.up1 = (Up(args.rfbdim *2, args.rfbdim))
        self.up2 = (Up(args.rfbdim *2, args.rfbdim))
        self.up3 = (Up(args.rfbdim *2, args.rfbdim))
        self.up4 = (Up(args.rfbdim *2, args.rfbdim))
        self.side1 = nn.Conv2d(args.rfbdim, 1, kernel_size=1)
        self.side2 = nn.Conv2d(args.rfbdim, 1, kernel_size=1)
        self.head = nn.Conv2d(args.rfbdim, 1, kernel_size=1)


    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')

        return out, out1, out2

class SAM2UNetDC(nn.Module):
    def __init__(self, model_cfg=None, checkpoint_path=None, args=None, device=None):
        super(SAM2UNetDC, self).__init__()
        #* DC
        target_H_W = (352,352)
        self.SPE = Simple_Point_Erosion_module(target_H_W=target_H_W)

        self.net = SAM2UNet(model_cfg, checkpoint_path, args, device)
        self.is_load = False

    def load_chpt_from_path(self, ckpt_path):
        self.net.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.is_load = True
    
    
    def Simple_Component_Erosion(self, T, M_T):
        """
        Simple Component Erosion
        T: binary structure, shape: 1,1,H,W
        M_T: the mask of specified region, binary,  shape 1,1,H,W
        """
        import skimage
        # compute connected components of M_T
        
        M_T_numpy = M_T.squeeze().detach().cpu().numpy().astype(np.uint8)
        Lcc = skimage.measure.label(M_T_numpy, connectivity=2)
        
        out, _ = self.SPE.RSPE(T=T, M_T=M_T, max_K=np.inf)
        remained = Lcc * out.squeeze().detach().cpu().numpy()
        remained_cc_label = np.unique(remained)
        
        result_array = (T - M_T).detach().squeeze().cpu().numpy()   # H,W
        
        # Retrieve
        for label in remained_cc_label:
            if label == 0:
                continue
            result_array = result_array + (Lcc == label) * 1
        
        result_array = torch.from_numpy(result_array).to(T.device)
        return result_array
    
    @torch.no_grad()
    def DeepDilation(self,T,is_infer_sliding_window=False, sw_roi_size=(224,224),sw_batch_size=4,verbose=False):
        """ DeepDilation
        using pre-trained AutoEncoder to indicate regions of disconnection
        T: presegmented image,  binary, shape: shape, B,1,H,W
        is_infer_sliding_window: whether to use sliding window inference
        verbose: whether to print the information
        sw_roi_size: the size of sliding window
        """
        
        self.eval()
        
        if is_infer_sliding_window:
            sw_batch_size = 4
            roi_size = sw_roi_size if sw_roi_size !=None else self.roi_size
            if verbose:
                print("using sliding window inference with roi_size:{}".format(roi_size))
            T_dd = sliding_window_inference(
                T, roi_size = roi_size, sw_batch_size= sw_batch_size, predictor= self.net.net)
        else:
            if verbose:
                print("not using sliding window inference".format(self.roi_size))
            T_dd= self.net(T)
        return T_dd
    
    @torch.no_grad()
    def DeepClosing(self,T,is_infer_sliding_window=True, sw_roi_size=(224,224),sw_batch_size=4,verbose=False):
        if type(T) == torch.Tensor:
            # assert T shape is B,1,H,W
            assert len(T.shape) == 4, "T shape must be B,1,H,W"
            # assert binay 
            #assert T.unique() == torch.tensor([0,1]), "T must be binary"
        print(T.shape)
        T_dd, _, _ = self.DeepDilation(T=T,is_infer_sliding_window=False, sw_roi_size=(224,224),sw_batch_size=4,verbose=False)
        T_dd = torch.sigmoid(T_dd)

        #TODO cubic
        #T_dd = transform.inverse(T_dd.squeeze().cpu())
        

        T_dd[T_dd >= 0.5] = 1
        T_dd[T_dd < 0.5] = 0
        #T_dd = AsDiscrete(threshold=0.2)(T_dd).int()

        # assert (T[:,0,:,:] == T[:,1,:,:]).all()
        # assert (T[:,0,:,:] == T[:,2,:,:]).all()

        T = T[:,0,:,:]

        M_T = T_dd.squeeze() - T.squeeze()
        M_T[M_T<0] =0
        M_T = M_T.squeeze()
        
        T_dc = self.Simple_Component_Erosion(T=T_dd,M_T=M_T.unsqueeze(0).unsqueeze(0))    
        
        return {"T_dd":T_dd,"T_dc":T_dc,"M_T":M_T,"T":T}


if __name__ == "__main__":

    parser = argparse.ArgumentParser("SAM2-UNet")
    
    parser.add_argument("--input_channel", type=int, default=3, help="input data channels")
    parser.add_argument("--dsc_number", type=int, default=16, help="dsc conv num")
    parser.add_argument("--adadim", type=int, default=32, help="adapter dim default to 32")
    parser.add_argument("--rfbdim", type=int, default=256, help="RFB module dim default to 256")

    args = parser.parse_args()

    device = torch.device("cuda:0")
    pretrain_path = '/root/shared-nvme/lyx/Project/Nerve/Trial/Pretrian/sam2_hiera_large.pt'
    test_volume = '/root/shared-nvme/lyx/Project/Nerve/Data/Sert-Stanford/test/test/volume-8bit/volume-2.tiff'

    volume = sitk.ReadImage(test_volume)
    volume = sitk.GetArrayFromImage(volume)


    with torch.no_grad():
        model = SAM2UNetDC(
            model_cfg='sam2_hiera_l',
            checkpoint_path=pretrain_path,
            args=args,
            device=device
        ).to(device)

        x = torch.randn(1, 3, 352, 352).cuda()
        res = model.DeepClosing(x)
        print(res)
