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
parser.add_argument('--nepochs', type=int, default=181, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

model_path = "checkpoint/netG_model_epoch_{}.pth".format(opt.nepochs)

#net_g = torch.load(model_path).to(device)
net_g = torch.load(model_path,map_location=torch.device('cpu')).to("cpu")

# if opt.direction == "a2b":
#    image_dir = "dataset/{}/test/a/".format(opt.dataset)
# else:
#    image_dir = "dataset/{}/test/b/".format(opt.dataset)

out_dir = "checkpoint_mip/result_16/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

script_path =os.getcwd()
image_dir = os.path.join(script_path, "../../ARIC/pd_wip/pd_nifti_final/pd_test")
image_filenames = [x for x in os.listdir(image_dir) if x.endswith('.gz')]

# transform_list = [transforms.ToTensor(),
#                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

# transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    volume1 = load_tiff_volume_and_scale_si(image_dir, image_name)
    volume1 = volume1.transpose(2,0,1)
    img = np.expand_dims(volume1,-1)
    img = img.transpose(0,3,1,2)
    input = torch.from_numpy(img)
    out = net_g(input)
    out_img = out.detach().squeeze(1).cpu()
    out_img = out_img.numpy().transpose(1,2,0)

    FileSave(out_img,out_dir+image_name)
