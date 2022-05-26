import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', ds=8):
        self._dims = dims[-2:]
        self.mode = mode
        self.ds = ds
        self.ht, self.wd = self._dims
        pad_ht = (((self.ht // ds) + 1) * ds - self.ht) % ds
        pad_wd = (((self.wd // ds) + 1) * ds - self.wd) % ds
        self._ht_pad = self.ht + pad_ht
        self._wd_pad = self.wd + pad_wd
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        inputs_padded = []
        for inp in inputs:
            if inp.shape[-2:] != self._dims:
                raise ValueError(f"Wrong dimensionality {inp.shape} should have H,W = {self._dims}")
            if inp.ndim == 4 :
                # Standard NCHW tensor
                inputs_padded += [F.pad(inp, self._pad, mode='replicate') ]
            else:
                # Non-Standard Tensor => Bring to 4D N1HW for padding
                inp4D = inp.reshape([-1,1,self.ht,self.wd])
                inp4D_pad = F.pad(inp4D, self._pad, mode='replicate') 
                shape_pad = inp.shape[:-2] + (self._ht_pad, self._wd_pad)
                inputs_padded += [inp4D_pad.reshape(shape_pad)]
        return inputs_padded

    def unpad(self,x):
        if x.shape[-2]!=self._ht_pad or x.shape[-1]!=self._wd_pad:
            raise ValueError(f"Wrong dimensionality {x.shape} should have padded shape H,W == {self._ht_pad},{self._wd_pad}")
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

    def __repr__(self):
        return f"InputPadder(dims={self.ht, self.wd}, mode='{self.mode}', ds={self.ds})"

def forward_interpolate(flow:torch.Tensor):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img:torch.Tensor, coords:torch.Tensor, mode:str='bilinear'):
    """ Wrapper for grid_sample, uses pixel coordinates """
    assert coords.shape[-1] == 2
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)

    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    return img

def bilinear_sampler_with_mask(img:torch.Tensor, coords:torch.Tensor, mode:str='bilinear'):
    """ Wrapper for grid_sample, uses pixel coordinates """
    assert coords.shape[-1] == 2
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    # only count valid if all 4 pixels (bilinear) are within the image space
    mask_valid = (xgrid > 0) & (ygrid > 0) & (xgrid < (W-1)) & (ygrid < (H-1))

    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    return img, mask_valid.float()

def coords_grid(batch:int, ht:int, wd:int):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack((coords[1],coords[0]), dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow:torch.Tensor, mode:str='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
