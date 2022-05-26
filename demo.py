import sys
sys.path.append('core')

import argparse
import os
from os.path import join
import glob
import numpy as np
import torch
from PIL import Image
from imageio import imsave

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time
from core.utils.flow_utils import flow_warp, est_occl_heu
from core.utils.frame_utils import writeFlowKITTI

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.stack([img,img,img],axis=-1)
    img = img[:,:,0:3].copy()
    img = torch.from_numpy(img).permute(2, 0, 1).float().contiguous()
    return img


def load_image_list(image_files):
    images = []
    for imfile in sorted(image_files):
        images.append(load_image(imfile))
 
    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]
        

class SimpleVisualizer(object):
    def __init__(self, shape, args):

        self.shape = shape
        self.args = args
        self.fwdbwd = args.fwdbwd
        self.occa1 = args.occa1
        self.occa2 = args.occa2
        self.showfig = args.showfig
        self.savefiles = args.savefiles
        self.stop = False
        self.build_fig()
        self.register()

    def build_fig(self):
        self.fig = plt.figure(num=10,clear=True)
        if self.fwdbwd:
            self.gs = self.fig.add_gridspec(2,5)
        else:
            self.gs = self.fig.add_gridspec(2,3)
        gs = self.gs
        # plot images on left side
        self.ax_I1  = self.fig.add_subplot(gs[1, 1])
        self.ax_I2  = self.fig.add_subplot(gs[0, 0])
        self.ax_I2w = self.fig.add_subplot(gs[0, 1])
        self.ax_flo = self.fig.add_subplot(gs[1, 0])
        self.ax_err = self.fig.add_subplot(gs[0, 2])
        self.ax_I1.set_title("I1")
        self.ax_I2.set_title("I2")
        self.ax_I2w.set_title("I2w")
        self.ax_flo.set_title("F1->2")
        self.ax_err.set_title("ErrRGB I1-I2w")
        if self.fwdbwd:
            self.ax_flobw = self.fig.add_subplot(gs[1, 2])
            self.ax_occfw = self.fig.add_subplot(gs[0, 3])
            self.ax_occbw = self.fig.add_subplot(gs[1, 3])
            self.ax_dflowfw = self.fig.add_subplot(gs[0, 4])
            self.ax_dflowbw = self.fig.add_subplot(gs[1, 4])
            self.ax_flobw.set_title("F2->1")
            self.ax_occfw.set_title("Occ1 F1-F2w")
            self.ax_occbw.set_title("Occ2 F2-F1w")
            self.ax_dflowfw.set_title("ErrFlo F1-F2w")
            self.ax_dflowbw.set_title("ErrFlo F2-F1w")

        self.ax_continue =  plt.axes([0.7, 0.05, 0.1, 0.075])
        self.fig.subplots_adjust(top=0.995,bottom=0.15, left=0.005,right=0.995,hspace=0.005,wspace=0.005)
        for ax in self.fig.get_axes():
            ax.set_xticks([])
            ax.set_yticks([])

    def on_next_callback(self, clbck):
        #print("nbext pressed", clbck)
        self.fig.canvas.stop_event_loop()  # event loop stopped => blocking stopped, program continues
    
    def on_close(self, event):
        self.stop = True
        self.fig.canvas.stop_event_loop()  # event loop stopped => blocking stopped, program continues

    def update(self, img1, img2, flo, names=("I1","I2")):
        img2w = flow_warp(img2, flo[0:1]) 
        img1_np = img1[0].permute(1,2,0).cpu().numpy()
        img2_np = img2[0].permute(1,2,0).cpu().numpy()
        img2w_np = img2w[0].permute(1,2,0).cpu().numpy()
        flofw_np = flo[0].permute(1,2,0).cpu().numpy()
        flofw_img_np = flow_viz.flow_to_image(flofw_np)

        if self.fwdbwd:
            img1w = flow_warp(img1, flo[1:2]) 
            img1w_np = img1w[0].permute(1,2,0).cpu().numpy()

            flobw_np = flo[1].permute(1,2,0).cpu().numpy()
            flobw_img_np = flow_viz.flow_to_image(flobw_np)

            flodelta_fw, occl_fw = est_occl_heu(flo[0:1], flo[1:2], self.occa1, self.occa2)
            flodelta_bw, occl_bw = est_occl_heu(flo[1:2], flo[0:1], self.occa1, self.occa2)

            occl_fw_np = occl_fw.cpu().numpy()[0]
            occl_bw_np = occl_bw.cpu().numpy()[0]

            flodelta_fw_np  = torch.norm(flodelta_fw,p=2,dim=1).cpu().numpy()[0]
            flodelta_bw_np  = torch.norm(flodelta_bw,p=2,dim=1).cpu().numpy()[0]

        if self.showfig:
            self.ax_I1 .imshow(img1_np.astype(np.uint8))
            self.ax_I2 .imshow(img2_np.astype(np.uint8))
            self.ax_I2w.imshow(img2w_np.astype(np.uint8))
            self.ax_flo.imshow(flofw_img_np.astype(np.uint8))
            self.ax_err.imshow(np.abs(img2w_np-img1_np).sum(axis=-1).astype(np.uint8))

            if self.fwdbwd:
                self.ax_flobw.imshow(flobw_img_np)
                self.ax_occfw.imshow(occl_fw_np*1.)
                self.ax_occbw.imshow(occl_bw_np*1.)
                self.ax_dflowfw.imshow(flodelta_fw_np)
                self.ax_dflowbw.imshow(flodelta_bw_np)
                        
            plt.ion()
            plt.show()
            self.fig.canvas.start_event_loop()  #starts a blocking event loop

        if self.savefiles:
            name1_str = os.path.splitext(os.path.basename(names[0]))[0]
            name2_str = os.path.splitext(os.path.basename(names[1]))[0]
            names_str = f"{name1_str}__{name2_str}"

            out_folder = join(self.args.out_path,"./out_png/")
            os.makedirs(out_folder, exist_ok=True)

            imsave(join(out_folder, f'{names_str}_01_img1.png'), img2_np.astype(np.uint8))
            imsave(join(out_folder, f'{names_str}_02_img0.png'), img1_np.astype(np.uint8))
            imsave(join(out_folder, f'{names_str}_03_fwd_warped.png'), img2w_np.astype(np.uint8))
            imsave(join(out_folder, f'{names_str}_04c_flowvis.png'), flofw_img_np.astype(np.uint8))
            imsave(join(out_folder, f'{names_str}_04_fwd_warp_err.png'), np.abs(img2w_np-img1_np).sum(axis=-1).astype(np.uint8))
            writeFlowKITTI(join(out_folder, f'{names_str}_04d_flowvec.png'), flofw_np )

            if self.fwdbwd:
                imsave(join(out_folder, f'{names_str}_04e_flowdelta.png'), flodelta_fw_np.astype(np.uint8))
                imsave(join(out_folder, f'{names_str}_04f_flowocclheu.png'), (occl_fw_np*255).astype(np.uint8))
                imsave(join(out_folder, f'{names_str}_05_img0.png'), img1_np.astype(np.uint8))
                imsave(join(out_folder, f'{names_str}_06_img1.png'), img2_np.astype(np.uint8))
                imsave(join(out_folder, f'{names_str}_07_bwd_warped_bwd.png'), img1w_np.astype(np.uint8))
                imsave(join(out_folder, f'{names_str}_08_bwd_warp_err_bwd.png'), np.abs(img1w_np-img2_np).sum(axis=-1).astype(np.uint8))
                imsave(join(out_folder, f'{names_str}_08c_flowvis_bwd.png'), flobw_img_np.astype(np.uint8))
                writeFlowKITTI(join(out_folder, f'{names_str}_08d_flowvec_bwd.png'), flobw_np) 
                imsave(join(out_folder, f'{names_str}_08e_flowdelta_bwd.png'), flodelta_bw_np.astype(np.uint8))
                imsave(join(out_folder, f'{names_str}_08f_flowocclheu_bwd.png'), (occl_bw_np*255).astype(np.uint8))     

    def register(self):
        self.bn_continue = Button(self.ax_continue, 'Continue')
        self.bn_continue.on_clicked(lambda x: self.on_next_callback(x) )
        
        self.close_fn = self.fig.canvas.mpl_connect('close_event', lambda x: self.on_close(x) )



def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    if not os.path.isfile(args.model):
        raise ValueError(f"Could not find model {args.model} ")
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        image_names = glob.glob(os.path.join(args.path, '*.png')) + \
                      glob.glob(os.path.join(args.path, '*.jpg'))
        image_names.sort()

        images = load_image_list(image_names)
        vis = SimpleVisualizer(images[0].shape, args)

        for i in range(images.shape[0]-1):
            image1 = images[i  ][None]
            image2 = images[i+1][None]
            t1 = time.time()
            flow_low, flow_up = model(image1, image2, iters=args.iters, test_mode=True)
            t2 = time.time()
            print("dt", (t2-t1)*1000)
            vis.update(image1, image2, flow_up, names=(image_names[i],image_names[i+1]))
            if vis.stop:
                print("stopping flow computation")
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--out_path', default="out/", help="file output path")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--fwdbwd', type=int,default=0, help='compute fwd bwd flow')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--iters', default=20, type=int, help='use efficent correlation implementation')
    parser.add_argument('--occa1', default=0.1, type=float, help='occl heuristic alpha1')
    parser.add_argument('--occa2', default=0.5, type=float, help='occl heuristic alpha2')
    parser.add_argument('--showfig', default=1, type=int, help='show figure')
    parser.add_argument('--savefiles', default=0, type=int, help='show figure')
    args = parser.parse_args()

    demo(args)
