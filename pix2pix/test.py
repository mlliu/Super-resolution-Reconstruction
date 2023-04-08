from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms

from data import *

from utils import is_image_file, load_img, save_img

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
#parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--norm_type', choices=['max','percentile'], default='max', help='normalization type')
parser.add_argument('--mip_type', default=False, help='mip type')
parser.add_argument('--modelfile',type = str,help="the path to save the model")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

#model path
modelname = opt.modelfile +"netG_model_epoch_{}.pth".format(opt.nepochs)
net_g = torch.load(modelname,map_location=torch.device('cpu')).to("cpu")

# generated image save path
out_dir = opt.modelfile+"result/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
script_path =os.getcwd()

#test image path
if opt.mip_type==16:
    testname = 'test16'
else:
    testname = 'test'
image_dir = os.path.join(script_path, "/home/mliu121/data-yqiao4/pd_wip/pd_nifti_final",testname)
image_filenames = [x for x in os.listdir(image_dir) if x.endswith('.gz')]

# transform_list = [transforms.ToTensor(),
#                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

# transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    volume1 = load_gz_to_numpy_vol(image_dir, image_name,norm_type=opt.norm_type)
    volume1 = volume1.transpose(2,0,1)
    img = np.expand_dims(volume1,-1)
    img = img.transpose(0,3,1,2)
    input = torch.from_numpy(img)
    out = net_g(input)
    out_img = out.detach().squeeze(1).cpu()
    out_img = out_img.numpy().transpose(1,2,0)

    FileSave(out_img,out_dir+image_name)
