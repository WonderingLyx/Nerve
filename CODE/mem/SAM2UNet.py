import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dataset import FullDataset

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
    
class MemoryBank:
    def __init__(self, num_maskmem:int=7):
        self.num_maskmem = num_maskmem
        self.bank = {}

    def add_mem(self, current_out:dict, frame_idx:int):
        if len(self.bank) > self.num_maskmem:
            _ = self.pop_mem()
        else:
            self.bank[frame_idx] = current_out

    def pop_mem(self):
        if len(self.bank) == 0:
            return None
        idx_sm = min(self.bank.keys())
        return self.bank.pop(idx_sm)
    
    def get(self, idx:int):
        return self.bank.get(idx, None)
        

    def clear(self):
        self.back = {}

    def __len__(self):
        
        return len(self.bank)

    def __str__(self):
       
        return str(self.bank)
    


class SAM2UNet(nn.Module):
    def __init__(self, model_cfg=None, checkpoint_path=None, args=None) -> None:
        super(SAM2UNet, self).__init__()   

        #* sam params
        self.num_feature_levels = 3 #使用三张特征图
        self.num_maskmem = args.get('num_maskmem', 7)
        self.bank = MemoryBank(self.num_maskmem)
        #self.max_cond_frames_in_attn = 2 #*最大存储的 condition frame num
        self.hidden_dim = 256
        self.sigmoid_scale_for_mem_enc = 20.0
        self.sigmoid_bias_for_mem_enc = -10.0
        self.run_mem_encoder = True
        self.is_init_cond_frame = True
        self.mem_dim = self.hidden_dim
        self.prompt_freq = args.get('prompt_freq', 2) # 每隔几个batch进行memory attention

        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.memory_temporal_stride_for_eval = 1
        
        self.directly_add_no_mem_embed = True #*第一帧是否直接跳过 memory encoder
        
        if model_cfg is None: 
            model_cfg = "sam2_hiera_l.yaml"
        else:
            sam_types = ['sam2_hiera_l', 'sam2_hiera_s']
            assert model_cfg in sam_types
            model_cfg = model_cfg + '.yaml'

        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder

        self.memory_encoder = model.memory_encoder
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
            self.memory_encoder.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(self.num_maskmem, 1, 1, self.mem_dim)
        )

        self.memory_attention = model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        self.neck = model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []

        #* 加入Adapter
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block, args.adapter)
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

    
    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes
    
    def select_closest_cond_frames(self, frame_idx, cond_frame_outputs, max_cond_frame_num):
        """
        Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
        that are temporally closest to the current frame at `frame_idx`. Here, we take
        - a) the closest conditioning frame before `frame_idx` (if any);
        - b) the closest conditioning frame after `frame_idx` (if any);
        - c) any other temporally closest conditioning frames until reaching a total
            of `max_cond_frame_num` conditioning frames.

        Outputs:
        - selected_outputs: selected items (keys & values) from `cond_frame_outputs`.
        - unselected_outputs: items (keys & values) not selected in `cond_frame_outputs`.
        """
        if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
            selected_outputs = cond_frame_outputs
            unselected_outputs = {}
        else:
            assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
            selected_outputs = {}

            # the closest conditioning frame before `frame_idx` (if any)

            #TODO 只选择之前的帧
            idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
            if idx_before is not None:
                selected_outputs[idx_before] = cond_frame_outputs[idx_before]

            # # the closest conditioning frame after `frame_idx` (if any)
            # idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
            # if idx_after is not None:
            #     selected_outputs[idx_after] = cond_frame_outputs[idx_after]

            # add other temporally closest conditioning frames until reaching a total
            # of `max_cond_frame_num` conditioning frames.
            num_remain = max_cond_frame_num - len(selected_outputs)
            inds_remain = sorted(
                (t for t in cond_frame_outputs if t not in selected_outputs),
                key=lambda x: abs(x - frame_idx),
            )[:num_remain]
            selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
            unselected_outputs = {
                t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
            }

        return selected_outputs, unselected_outputs
    
    def _prepare_memory_conditioned_features(
            self,
            frame_idx,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            track_in_reverse=False
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device

        #* 不融入memory
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        if not self.is_init_cond_frame:
            to_cat_memory, to_cat_memory_pos_embed = [], []
            assert len(self.bank) > 0

            #* 选择提示了的帧,加入了click or box or mask, 任务中不引入prompts
            # selected_cond_outputs, unselected_cond_outputs = self.select_closest_cond_frames(
            #     frame_idx, cond_outputs, self.max_cond_frames_in_attn
            # )
            # t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            
            #* 选择当前帧之前的帧
            t_pos_and_prevs = []
            r = self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // r) * r
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // r) * r
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * r
                out = self.bank.get(prev_frame_idx)
                # if out is None:
                #     # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                #     # frames, we still attend to it as if it's a non-conditioning frame.
                #     out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))
            
            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].cuda(non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].cuda()
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)
        else: 
            if self.directly_add_no_mem_embed: #* 其他帧是否跳过 memory encoder 
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid emtpy memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=0,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem


    def track_step(
            self, 
            frame_idx,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            run_mem_encoder=True
    ):
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        
        #! fused the visual feature with previous memory features in the memory bank

        pix_feat_with_mem = self._prepare_memory_conditioned_features(
            frame_idx,
            current_vision_feats=current_vision_feats[-1:],
            current_vision_pos_embeds=current_vision_pos_embeds[-1:],
            feat_sizes=feat_sizes[-1:],
        )
        return pix_feat_with_mem
    

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
       
        # scale the raw mask logits with a temperature before applying sigmoid
        #* 是否将mask 二值化
        
        # apply sigmoid on the raw mask logits to turn them into range (0, 1)
        mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        return maskmem_features, maskmem_pos_enc
        
    
    def forward(self, x, img_idx:int=0, batch_idx:int=0, batch_size:int=2):
        x1, x2, x3, x4 = self.encoder(x) #sam.trunk
        feats, pos = self.neck([x1,x2,x3,x4])

        backbone_out = {}
        backbone_out['backbone_fpn'] = feats
        backbone_out['vision_pos_enc'] = pos
        
        (
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        #* memory interference
        
        if ((batch_idx+1) % self.prompt_freq) == 0:
            self.is_init_cond_frame = False
        else:
            self.is_init_cond_frame = True

        pix_feat_with_mem = self.track_step(
            batch_idx,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes
        )
        
        #* decoder part
        (out, out1, out2), (mask) = self.decoder(x1, x2, x3, pix_feat_with_mem)
        
        current_out = {}
        if self.run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = mask
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        self.bank.add_mem(current_out=current_out, frame_idx=batch_idx)



        return out, out1, out2
    
    def decoder(self, x1,x2,x3,x4):
        
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), x4
        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        #mask1 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')

        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        #mask2 = F.interpolate(self.side2(x), scale_factor=4, mode='bilinear')

        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        mask = F.interpolate(self.head(x), scale_factor=2, mode='bilinear')

        return (out, out1, out2), (mask)




if __name__ == "__main__":

    config_path = '/mnt/40B2A1DBB2A1D5A6/lyx/project/MedSam2/SAM-Unet/CONFIG/SAM2-Unet_Mem.yaml'
    config = OmegaConf.load(config_path)
    
    args = config.model
    device = torch.device("cuda")

    args = config.dataset
    dataset_1 = FullDataset(size=352, mode='train', view='axial', **args)
    #dataset_2 = FullDataset(size=352, mode='train', view='coronal', **args)
    dataloader_1 = DataLoader(dataset_1, batch_size=args.batch_size, shuffle=False, num_workers=8)
    #dataloader_2 = DataLoader(dataset_2, batch_size=args.batch_size, shuffle=False, num_workers=8)

    args = config.model
    with torch.no_grad():
        model = SAM2UNet(args.sam_type, args.hiera_path, args.args).to(device)
        for batch_id, batch in enumerate(dataloader_1):
            x, target_1 = batch['image'].to(device), batch['label'].to(device)
            image_index = batch['image_index'][0]
            out, out1, out2 = model(x,image_index,batch_id,config.dataset.batch_size)
            print(out.shape, out1.shape, out2.shape)

