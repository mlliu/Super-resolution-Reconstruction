from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data import get_dataset,get_training_set,get_test_set
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
#parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=16, help='testing batch size')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
parser.add_argument('--ngf', type=int, default=32, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=32, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=2, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--debug', default=False, help='debug mode')
parser.add_argument('--norm_type', choices=['max','percentile'], default='max', help='normalization type')
parser.add_argument('--mip_type', type = int, default=0, help='mip type')
parser.add_argument('--modelfile',type = str,help="the path to save the model")

opt = parser.parse_args()

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

# +
print('===> Loading datasets')
root_path = "dataset/"
xtrain,ytrain,xtest,ytest = get_dataset(opt.debug,norm_type=opt.norm_type,mip_type=opt.mip_type)
train_set = get_training_set(xtrain,ytrain)
test_set = get_test_set(xtest,ytest)

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)
# -

device = torch.device("cuda:0" if opt.cuda else "cpu")
print("device is: ",device)

print('===> Building models')
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)

print('===> setting the loss')
criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

# setup optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

#model path to save
#modelfile = "checkpoint_norm_" + opt.norm_type + "_mip_ " + opt.mip_type
if not os.path.exists(opt.modelfile):
    os.mkdir(opt.modelfile)

print('===> Training')
g_loss = []
d_loss = []
test_loss = []
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    print("epoch: ",epoch )
    for iteration, batch in enumerate(training_data_loader, 1):
        print("iteration", iteration)
        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b = net_g(real_a)

        ######################
        # (1) Update D network
        ######################
        
        optimizer_d.zero_grad()
        
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        #print("loss step 1")
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        #print("loss step 2")
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
       
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
        
        loss_g = loss_g_gan + loss_g_l1
        
        loss_g.backward()

        optimizer_g.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))
        g_loss.append(loss_g.item())
        d_loss.append(loss_d.item())
    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    # test
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = batch[0].to(device), batch[1].to(device)

        prediction = net_g(input)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    test_loss.append(avg_psnr / len(testing_data_loader))

    #checkpoint
    if epoch % 20 == 0:

        #if not os.path.exists(os.path.join("checkpoint","pd_wip")):
        #    os.mkdir(os.path.join("checkpoint", "pd_wip"))
        net_g_model_out_path = opt.modelfile + \
                               "netG_model_epoch_{}.pth".format(epoch)
        net_d_model_out_path = opt.modelfile+ \
                               "netD_model_epoch_{}.pth".format(epoch)
        torch.save(net_g, net_g_model_out_path)
        torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint"))

#after finishing training
#plot loss curve based on g_loss, d_loss and test_loss and save the figure to the modelpath
plt.figure()
plt.plot(g_loss, label='g_loss')
plt.plot(d_loss, label='d_loss')
plt.plot(test_loss, label='test_loss')
plt.legend()
plt.savefig(opt.modelfile + "loss_curve.png")
