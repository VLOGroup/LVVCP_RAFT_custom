import numpy as np

import torch
import torchvision.transforms.functional as TTF
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

for H,W in [(127,255),(128,256)]:
    C=3

    img = np.random.rand(H,W,C)
    ht, wd = img.shape[:2]
    scale_x = scale_y = np.random.rand()*0.6 + 0.7

    hn, wn = int(H*scale_y), int(W*scale_x)
    scale_y, scale_x = hn/ht, wn/wd
    img_ttf = TTF.resize(torch.from_numpy(img.transpose(2,0,1)), size=[hn,wn], interpolation=TTF.Image.BILINEAR).numpy().transpose(1,2,0)
    img_cv2 = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

    print("delta:", np.abs(img_ttf - img_cv2).mean())
    print("types:", img_ttf.dtype, img_cv2.dtype)
