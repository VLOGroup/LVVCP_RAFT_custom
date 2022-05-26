import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .corr import CorrBlock, AlternateCorrBlock
from .utils.utils import bilinear_sampler, bilinear_sampler_with_mask, coords_grid, upflow8

from typing import Dict, Tuple, List, Optional

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        self.fwdbwd:bool = args.fwdbwd if hasattr(args,"fwdbwd") else False

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        self.alternate_corr:bool = False  
        if 'alternate_corr' in self.args:
            self.alternate_corr = args.alternate_corr

        self.mixed_precision:bool = False
        if 'mixed_precision' in self.args:
            self.mixed_precision = self.args.mixed_precision

        self.corr_radius:int = args.corr_radius

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img:torch.Tensor):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        # print (img.shape, img.device)
        coords0 = coords_grid(N, H//8, W//8)
        coords1 = coords_grid(N, H//8, W//8)
        # print (coords0)
        coords0 = coords0.to(img.device, dtype=img.dtype)
        coords1 = coords1.to(img.device, dtype=img.dtype)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow:torch.Tensor, mask:torch.Tensor):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1:torch.Tensor, image2:torch.Tensor, iters:int=12, flow_init:Optional[torch.Tensor]=None, test_mode:bool=False, float_imgs01:bool=False) -> List[torch.Tensor]:
        """ Estimate optical flow between pair of frames """

        if float_imgs01:  # Floating point images in range [0..1]
            image1 = 2 * (image1) - 1.0
            image2 = 2 * (image2) - 1.0
        else:
            image1 = 2 * (image1 / 255.0) - 1.0
            image2 = 2 * (image2 / 255.0) - 1.0
        #import matplotlib; import matplotlib.pyplot as plt;matplotlib.use("Qt5Agg"); plt.imshow(image1[0,0].cpu(),'gray'); plt.show()

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        if self.mixed_precision:
            fmap1, fmap2 = self.run_fnet_w_autocast(image1, image2)
        else:
            fmap1, fmap2 = self.fnet([image1, image2])    

        if self.fwdbwd:
            fmap1, fmap2 = torch.cat([fmap1,fmap2],dim=0), torch.cat([fmap2,fmap1],dim=0)
            image12 = torch.cat([image1,image2],dim=0)
        else:
            image12 = image1   

        fmap1 = fmap1.to(dtype=image1.dtype)
        fmap2 = fmap2.to(dtype=image1.dtype)
        # if self.alternate_corr:
        #     corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.corr_radius)
        # else:
        #     corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # run the context network
        if self.mixed_precision:
            net_hid, inp_feats = self.run_context_w_autocast(image12, hdim, cdim)
        else:
            net_hid, inp_feats = self.run_context_wo_autocast(image12, hdim, cdim)

        coords0, coords1 = self.initialize_flow(image12)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        assert iters > 0
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            if self.mixed_precision:
                net_hid, up_mask, delta_flow = self.run_updated_block_w_autocast(net_hid, inp_feats, corr, flow)
            else:
                net_hid, up_mask, delta_flow = self.update_block(net_hid, inp_feats, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            flow_low = coords1 - coords0
            ret = [flow_low, flow_predictions[-1]]
        else:
            ret = flow_predictions

        return ret

    @torch.jit.unused
    def run_context_w_autocast(self, image12:torch.Tensor, hdim:int, cdim:int) -> Tuple[torch.Tensor,torch.Tensor]:
        """ Simple wrappers to allow torch autocast if torchscript is not used """
        with autocast(enabled=self.mixed_precision):
            return self.run_context_wo_autocast(image12, hdim, cdim)

    def run_context_wo_autocast(self, image12:torch.Tensor, hdim:int, cdim:int) -> Tuple[torch.Tensor,torch.Tensor]:
        """ Simple wrappers to allow remove autocast if torchscript is used """
        cnet = self.cnet([image12])
        net_hid, inp_feats = torch.split(cnet[0], [hdim, cdim], dim=1)
        net_hid = torch.tanh(net_hid)
        inp_feats = torch.relu(inp_feats)
        return net_hid, inp_feats

    @torch.jit.unused
    def run_updated_block_w_autocast(self, net_hid, inp_feats, corr, flow):
        """ Simple wrappers to allow torch autocast if torchscript is not used """
        with autocast(enabled=self.mixed_precision):
            return self.update_block(net_hid, inp_feats, corr, flow)

    @torch.jit.unused
    def run_fnet_w_autocast(self, image1, image2):
        """ Simple wrappers to allow torch autocast if torchscript is not used """
        with autocast(enabled=self.mixed_precision):
            return self.fnet([image1, image2])    