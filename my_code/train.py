import copy
import os
import torch
import argparse
import itertools
import numpy as np
from torch.utils import data
from torchvision import models
from unet import Unet
from my_unet import my_UNet
from tqdm import tqdm
import torch.optim as optim
from diffusion import GaussianDiffusion
from torchvision.utils import save_image
from utils import get_named_beta_schedule
from embedding import EmbedFC
from Scheduler import GradualWarmupScheduler
from my_dataloader import My_Dataset,transback
from diffusion import EMA


def train(params:argparse.Namespace):

    # set device
    device = torch.device("cuda", 1)

    sample_save_dir = (params.samdir + '/%s_diffusion_version1_dim%d_mask%.2f_b%d_w%.1f_iter%d/' %
                       (params.image_name[:-4],params.modch,params.mask_ratio,params.batchsize,params.w,params.iteration))
    model_save_dir = (params.moddir + '/%s_diffusion_version1_dim%d_mask%.2f_b%d_w%.1f_iter%d/' %
                       (params.image_name[:-4],params.modch,params.mask_ratio,params.batchsize,params.w,params.iteration))

    # load data
    dataset = My_Dataset(params.folder, params.image_name,params.mask_ratio,params.iteration)
    dataloader = data.DataLoader(dataset, batch_size=params.batchsize, shuffle=True, pin_memory=True)
    dataloader2 = data.DataLoader(dataset, batch_size=params.genbatch, shuffle=False)
    # initialize models
    # net = Unet(
    #             in_ch = params.inch,
    #             mod_ch = params.modch,
    #             out_ch = params.outch,
    #             ch_mul = params.chmul,
    #             num_res_blocks = params.numres,
    #             cdim = params.cdim,
    #             use_conv = params.useconv,
    #             droprate = params.droprate,
    #             dtype = params.dtype
    #         )
    net = my_UNet(in_channels=params.inch,out_channels=params.outch,dim=params.modch,n_steps=params.T)
    # 原始设计使用
    # cemblayer = EmbedFC(input_dim=1000,emb_dim=256).to(device)
    resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    resnet.to(device)
    resnet.eval()
    # load last epoch
    lastpath = os.path.join(params.moddir,'last_epoch.pt')
    if os.path.exists(lastpath):
        lastepc = torch.load(lastpath)['last_epoch']
        # load checkpoints
        checkpoint = torch.load(os.path.join(params.moddir, f'ckpt_{lastepc}_checkpoint.pt'), map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        # cemblayer.load_state_dict(checkpoint['cemblayer'])
    else:
        lastepc = 0
    betas = get_named_beta_schedule(num_diffusion_timesteps = params.T)
    diffusion = GaussianDiffusion(
                    dtype = params.dtype,
                    model = net,
                    betas = betas,
                    w = params.w,
                    v = params.v,
                    device = device
                )

    # optimizer settings
    optimizer = torch.optim.AdamW(
                    itertools.chain(
                        diffusion.model.parameters(),
                        # cemblayer.parameters()
                    ),
                    lr = params.lr,
                    weight_decay = 1e-4
                )
    
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer = optimizer,
                            T_max = params.epoch,
                            eta_min = 0,
                            last_epoch = -1
                        )
    warmUpScheduler = GradualWarmupScheduler(
                            optimizer = optimizer,
                            multiplier = params.multiplier,
                            warm_epoch = params.epoch // 10,
                            after_scheduler = cosineScheduler,
                            last_epoch = lastepc
                        )
    ema = EMA(beta=0.995)
    ema_diffusion = copy.deepcopy(diffusion)

    if lastepc != 0:
        optimizer.load_state_dict(checkpoint['optimizer'])
        warmUpScheduler.load_state_dict(checkpoint['scheduler'])
    # training
    for epc in range(lastepc, params.epoch):
        # turn into train mode
        diffusion.model.train()
        # cemblayer.train()
        # batch iterations
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for i, (img, lab, pos) in enumerate(tqdmDataLoader):
                b = img.shape[0]
                optimizer.zero_grad()
                x_0 = img.to(device)
                lab = lab.to(device)
                pos = pos.to(device)
                cemb = resnet(lab)
                # cemb = cemblayer(cemb)
                cemb[np.where(np.random.rand(b)<params.threshold)] = 0
                loss = diffusion.trainloss(x_0, cemb = cemb, pos=pos)
                loss.backward()
                optimizer.step()
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "epoch": epc + 1,
                        "loss: ": loss.item(),
                        "batch per device: ":x_0.shape[0],
                        "img shape: ": x_0.shape[1:],
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    }
                )
                if i % params.update_ema_every == 0:
                    if i < params.step_start_ema:
                        ema_diffusion.load_state_dict(diffusion.state_dict())
                    ema.update_model_average(ema_diffusion, diffusion)
        warmUpScheduler.step()
        # evaluation and save checkpoint
        if (epc + 1) % params.interval == 0:
            ema_diffusion.model.eval()
            # cemblayer.eval()
            # generating samples
            # The model generate 80 pictures(8 per row) each time
            # pictures of same row belong to the same class
            with torch.no_grad():
                _, cond, pos = next(iter(dataloader2))
                cond = cond.to(device)
                pos = pos.to(device)
                cemb = resnet(cond)
                # cemb = cemblayer(cemb)
                genshape = (params.genbatch , *(_.shape[1:]))
                if params.ddim:
                    generated = ema_diffusion.ddim_sample(genshape, params.num_steps, params.eta, params.select, cemb = cemb, pos=pos)
                else:
                    generated = ema_diffusion.sample(genshape, cemb = cemb, pos=pos)
                samples_image = torch.clamp(transback(generated),0,1)

                if not os.path.exists(sample_save_dir):
                    os.makedirs(sample_save_dir)
                save_image(samples_image, os.path.join(sample_save_dir, f'generated_{epc+1}_{params.image_name[:-4]}.png'), nrow = 4)
            # save checkpoints
            checkpoint = {
                                'net':ema_diffusion.model.state_dict(),
                                # 'cemblayer':cemblayer.module.state_dict(),
                                'optimizer':optimizer.state_dict(),
                                'scheduler':warmUpScheduler.state_dict()
                            }
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            torch.save({'last_epoch':epc+1}, os.path.join(model_save_dir,'last_epoch.pt'))
            torch.save(checkpoint, os.path.join(model_save_dir, f'ckpt_{epc+1}_{params.image_name[:-4]}.pt'))
        torch.cuda.empty_cache()

def main():
    # several hyperparameters for model
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--batchsize',type=int,default=40,help='batch size per device for training Unet model')
    parser.add_argument('--numworkers',type=int,default=4,help='num workers for training Unet model')
    parser.add_argument('--inch',type=int,default=3,help='input channels for Unet model')
    parser.add_argument('--modch',type=int,default=128,help='model channels for Unet model')
    parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
    parser.add_argument('--outch',type=int,default=3,help='output channels for Unet model')
    parser.add_argument('--chmul',type=list,default=[1,2,2,2],help='architecture parameters training Unet model')
    parser.add_argument('--numres',type=int,default=2,help='number of resblocks for each block in Unet model')
    parser.add_argument('--cdim',type=int,default=256,help='dimension of conditional embedding')
    parser.add_argument('--useconv',type=bool,default=True,help='whether use convlution in downsample')
    parser.add_argument('--droprate',type=float,default=0.1,help='dropout rate for model')
    parser.add_argument('--dtype',default=torch.float32)
    parser.add_argument('--lr',type=float,default=2e-4,help='learning rate')
    parser.add_argument('--w',type=float,default=0.5,help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v',type=float,default=0.3,help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--epoch',type=int,default=30,help='epochs for training')
    parser.add_argument('--multiplier',type=float,default=2.5,help='multiplier for warmup')
    parser.add_argument('--threshold',type=float,default=0.1,help='threshold for classifier-free guidance')
    parser.add_argument('--interval',type=int,default=4,help='epoch interval between two evaluations')
    parser.add_argument('--moddir',type=str,default='model',help='model addresses')
    parser.add_argument('--samdir',type=str,default='sample',help='sample addresses')
    parser.add_argument('--genbatch',type=int,default=16,help='batch size for sampling process')
    parser.add_argument('--num_steps',type=int,default=100,help='sampling steps for DDIM')
    parser.add_argument('--eta',type=float,default=0,help='eta for variance during DDIM sampling process')
    parser.add_argument('--select',type=str,default='linear',help='selection stragies for DDIM')
    parser.add_argument('--ddim',type=lambda x:(str(x).lower() in ['true','1', 'yes']),default=False,help='whether to use ddim')
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
    parser.add_argument('--folder', type=str,default='/home/xianlong/图片/')
    parser.add_argument('--image_name', type=str,default='zebra.png')
    parser.add_argument('--mask_ratio', type=float,default='0.8')
    parser.add_argument('--iteration', type=int,default=5000)
    parser.add_argument('--update_ema_every',type=int,default=3)
    parser.add_argument('--step_start_ema',type=int, default=50)

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
