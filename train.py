import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import gc
import sys
import datetime

cudnn.benchmark = True  # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

parser = argparse.ArgumentParser(description='Normal-Aware MVSNet')
parser.add_argument('--mode', default='train', help='train or val', choices=['train', 'val', 'profile'])
parser.add_argument('--model', default='na_mvsnet', help='select model')

parser.add_argument('--dataset', default='eth', help='select dataset')
parser.add_argument('--trainpath', default='datasets/MVS_dataset', help='train datapath')
parser.add_argument('--valpath', default='datasets/MVS_dataset', help='val datapath')
# parser.add_argument('--trainlist', help='train list')
# parser.add_argument('--vallist', help='val list')

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2",
                    help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=64, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=20, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

# hyperparameters for loss
parser.add_argument('--alpha', type=float, default=0.1, help='coefficient for src images')
parser.add_argument('--beta', type=float, default=3.0, help='coefficient for normal maps')
parser.add_argument('--gamma', type=float, default=0.1, help='consistency loss between normal and depth')

# mindepth
parser.add_argument('--min_depth', type=float, default=1.4, help='min depth for camera')

# parse arguments and check
args = parser.parse_args()
if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None
if args.valpath is None:
    args.valpath = args.trainpath

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# create logger for mode "train" and "valall"
if args.mode == "train":
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)

# dataset, dataloader
MVSDataset = find_dataset_def(args.dataset)

train_dataset = MVSDataset(args.trainpath, "train", 5, args.numdepth, args.interval_scale)
val_dataset = MVSDataset(args.valpath, "val", 5, args.numdepth, args.interval_scale)

TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
valImgLoader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)
print('------+++++++------')
# model, optimizer
if args.model == 'na_mvsnet':
    model = NA_MVSNet(args.min_depth)
else:
    raise ValueError("model doesn't exist. Must be na_mvsnet")

if args.mode in ["train", "val"]:
    model = nn.DataParallel(model)

model.cuda()

model_loss = na_mvsnet_loss

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

# load parameters
start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "val" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the laval checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# main function
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print(
                'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TrainImgLoader), loss,
                                                                                     time.time() - start_time))

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))

        # valing
        avg_val_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(valImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = val_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'val', scalar_outputs, global_step)
                save_images(logger, 'val', image_outputs, global_step)
            avg_val_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, val loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                    len(valImgLoader), loss,
                                                                                    time.time() - start_time))
        save_scalars(logger, 'fullval', avg_val_scalars.mean(), global_step)
        print("avg_val_scalars:", avg_val_scalars.mean())
        lr_scheduler.step()
        # gc.collect()


def val():
    avg_val_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(valImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = val_sample(sample, detailed_summary=True)
        avg_val_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        print('Iter {}/{}, val loss = {:.3f}, time = {:3f}'.format(batch_idx, len(valImgLoader), loss,
                                                                   time.time() - start_time))
        if batch_idx % 100 == 0:
            print("Iter {}/{}, val results = {}".format(batch_idx, len(valImgLoader), avg_val_scalars.mean()))
    print("final", avg_val_scalars)


def train_sample(sample, detailed_summary=False):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth_ref"]
    mask = sample_cuda["mask_ref"]

    imgs_ref = sample_cuda["imgs_ref"].unsqueeze(1)  # [B, 1, h, w, 3]
    imgs_src = sample_cuda["imgs_src"]  # [B, N - 1, h, w, 3]
    imgs = torch.cat((imgs_ref, imgs_src), dim=1)

    proj_mat = sample_cuda['proj_mat']
    i_inv = sample_cuda['intrinsics_inv']
    depth_values = sample_cuda['depth_values']

    outputs = model(imgs, proj_mat, i_inv, depth_values)

    depth_est = outputs["depth"]
    normal_est = outputs['normal']

    loss = model_loss(sample_cuda, depth_est, normal_est, args)
    loss.backward()
    optimizer.step()

    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_est * mask.float(), "depth_gt": sample["depth_ref"]}

    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
        scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
        scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
        scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


@make_nograd_func
def val_sample(sample, detailed_summary=True):
    model.eval()
    with torch.no_grad():
        sample_cuda = tocuda(sample)
        depth_gt = sample_cuda["depth_ref"]
        mask = sample_cuda["mask_ref"]

        imgs_ref = sample_cuda["imgs_ref"].unsqueeze(1)  # [B, 1, h, w, 3]
        imgs_src = sample_cuda["imgs_src"]  # [B, N - 1, h, w, 3]
        imgs = torch.cat((imgs_ref, imgs_src), dim=1)

        proj_mat = sample_cuda['proj_mat']
        i_inv = sample_cuda['intrinsics_inv']
        depth_values = sample_cuda['depth_values']
        outputs = model(imgs, proj_mat, i_inv, depth_values)

        depth_est = outputs["depth"]
        normal_est = outputs['normal']

        loss = model_loss(sample_cuda, depth_est, normal_est, args)

        scalar_outputs = {"loss": loss}
        image_outputs = {"depth_est": depth_est * mask, "depth_gt": sample["depth_ref"]}

    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask

    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
    scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
    scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
    scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


def profile():
    warmup_iter = 5
    iter_dataloader = iter(valImgLoader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        val_sample(next(iter_dataloader), detailed_summary=True)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    if args.mode == "train":
        train()
    elif args.mode == "val":
        val()
    elif args.mode == "profile":
        profile()
