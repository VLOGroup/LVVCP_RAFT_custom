# Based on  flow_warp from: HD3  (which comes under ad BSD licence)
#    https://github.com/ucbdrive/hd3/blob/master/models/hd3_ops.py


# BSD 3-Clause License
#
# Copyright (c) 2019, Berkeley DeepDrive
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import torch
import torch.nn.functional as F

def flow_warp_pt_wrapped_np(x, flo):
    assert len(x.shape) ==3
    assert len(flo.shape) ==3
    assert flo.shape[-1]== 2
    assert flo.shape[0:2] == x.shape[0:2]
    x_th = torch.from_numpy(  x.transpose(2,0,1))[None,...].float()
    f_th = torch.from_numpy(flo.transpose(2,0,1))[None,...].float()
    xw_th = flow_warp(x_th, f_th)
    xw = xw_th.cpu().numpy().transpose(1,2,0)
    return xw

def flow_warp_with_mask(x:torch.Tensor, flo:torch.Tensor, align_corners:bool=True):
    """
    inverse warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    returns warped_output, valid_mask

    """
    if not torch.jit.is_scripting():
        assert type(x) == type(flo) == torch.Tensor, f"only implemented for torch tensors"
    assert flo.shape[1] == 2, f"flow shape must be N2HW but is {flo.shape}"
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(x.device, dtype=x.dtype)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(x.device, dtype=x.dtype)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1)

    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid = torch.stack([
        2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0,
        2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    ],
                        dim=1)

    vgrid = vgrid.permute(0, 2, 3, 1)
    # use same align_corners mode as in bilinear_sample used for sampling from cost volume in RAFT
    output = F.grid_sample(x, vgrid, padding_mode='border', align_corners=align_corners)
    valid_mask = torch.ones_like(x)
    valid_mask = F.grid_sample(valid_mask, vgrid, padding_mode='zeros', align_corners=align_corners)

    valid_mask[valid_mask < 0.9999] = 0  # allow a distance due to numerical errors.
    valid_mask[valid_mask > 0] = 1

    return output, valid_mask

def flow_warp(x:torch.Tensor, flo:torch.Tensor, align_corners:bool=True):
    warp, valid_mask = flow_warp_with_mask(x, flo, align_corners=align_corners)
    return warp * valid_mask

