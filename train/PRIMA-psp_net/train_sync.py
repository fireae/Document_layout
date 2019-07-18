import datetime, os, random, cv2
from PIL import Image

import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader

import sys
sys.path.append("../../")
import numpy as np
import utils.transforms as extended_transforms
import utils.joint_transforms as joint_transforms
from datasets import PRIMA
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d

import os
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"


train_args = {
    'train_args['num_classes']': 3,
    'epoch_num': 50,
    'lr': 1e-2,
    'longer_size': 1500,
    'crop_size': 700,
    'stride_rate': 1 / 2.,
    'weight_decay': 1e-3,
    'momentum': 0.95,
    'snapshot': '',  # empty string denotes learning from scratch
    'print_freq': 1,
    'val_save_to_img_file': False,
    'val_img_sample_rate': 0.1  # randomly sample some validation results to display
}
ckpt_path = '../../ckpt'
exp_name = 'PRIMA-pspnet-balanced'
writer = SummaryWriter(os.path.join(ckpt_path, 'exp', exp_name))

torch.cuda.set_device(args.local_rank)
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)

def init_palette():
    palette = [0,0,0, 64,128,64, 128,0,192]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    return palette

palette = init_palette()

def main(train_args):
    torch.cuda.set_device(args.local_rank)
    device = torch.cuda.set_device(args.local_rank)

    world_size = args.ngpu
    torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
    world_size=world_size,
    rank=args.local_rank,
    )

    net = PSPNet(train_args['num_classes'])
    # net = nn.DataParallel(PSPNet(train_args['num_classes'])).cuda()
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = net.to(device)
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
    )

    if len(train_args['snapshot']) == 0:
        curr_epoch = 1
        train_args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    else:
        print('training resumes from ' + train_args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, train_args['snapshot'])))
        split_snapshot = train_args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        train_args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                                     'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                                     'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11])}

    net.train()
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_joint_transform = joint_transforms.Compose([
        joint_transforms.Scale(train_args['longer_size']),
        joint_transforms.RandomCrop((train_args['longer_size'], train_args['longer_size'])),
        joint_transforms.RandomHorizontallyFlip()
    ])
    val_joint_transform = joint_transforms.Compose([
    joint_transforms.Scale(train_args['longer_size']),
    ])

    sliding_crop = joint_transforms.SlidingCrop(train_args['crop_size'], train_args['stride_rate'], ignore_label=255)

    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    restore_transform = standard_transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage(),
    ])
    visualize = standard_transforms.Compose([
        standard_transforms.Scale(400),
        standard_transforms.CenterCrop(400),
        standard_transforms.ToTensor()
    ])

    train_set = PRIMA('train', joint_transform=train_joint_transform, sliding_crop=sliding_crop,
                    transform=input_transform, target_transform=target_transform)
    val_set = PRIMA('val', sliding_crop=sliding_crop,
                    transform=input_transform, target_transform=target_transform)
    sampler = torch.utils.data.distributed.DistributedSampler(
    train_set,
    num_replicas=config.ngpu,
    rank=local_rank,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
    print(len(train_loader))

    # solved unbalanced distribution
    # weights = [1/6, 2/6, 3/6]
    # cls_weights = torch.FloatTensor(weights).cuda()
    # criterion = CrossEntropyLoss2d(weight=cls_weights, size_average=True).cuda()
    criterion = CrossEntropyLoss2d(size_average=True).cuda()
    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * train_args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': train_args['lr'], 'weight_decay': train_args['weight_decay']}
    ], betas=(train_args['momentum'], 0.999))

    if len(train_args['snapshot']) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_' + train_args['snapshot'])))
        optimizer.param_groups[0]['lr'] = 2 * train_args['lr']
        optimizer.param_groups[1]['lr'] = train_args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(train_args) + '\n\n')

    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=train_args['lr_patience'], min_lr=1e-10, verbose=True)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.1)
    for epoch in range(curr_epoch, train_args['epoch_num'] + 1):
        # validate(val_loader, net, criterion, optimizer, epoch, train_args, restore_transform, visualize)

        train(train_loader, net, criterion, optimizer, epoch, train_args)
        if epoch%5==0:
            validate(val_loader, net, criterion, optimizer, epoch, train_args, restore_transform, visualize)
        scheduler.step()


def train(train_loader, net, criterion, optimizer, epoch, train_args):
    train_main_loss = AverageMeter()
    train_aux_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        inputs, gts, _ = data
        assert len(inputs.size()) == 5 and len(gts.size()) == 4
        inputs.transpose_(0, 1)
        gts.transpose_(0, 1)

        assert inputs.size()[3:] == gts.size()[2:]
        slice_batch_pixel_size = inputs.size(1) * inputs.size(3) * inputs.size(4)

        for inputs_slice, gts_slice in zip(inputs, gts):
            inputs_slice = Variable(inputs_slice).cuda()
            gts_slice = Variable(gts_slice).cuda()

            optimizer.zero_grad()
            outputs, aux = net(inputs_slice)
            assert outputs.size()[2:] == gts_slice.size()[1:]
            assert outputs.size()[1] == train_args['num_classes']

            main_loss = criterion(outputs, gts_slice)
            aux_loss = criterion(aux, gts_slice)
            loss = main_loss + 0.4 * aux_loss
            loss.backward()
            optimizer.step()

            train_main_loss.update(main_loss.data[0], slice_batch_pixel_size)
            train_aux_loss.update(aux_loss.data[0], slice_batch_pixel_size)

        curr_iter += 1
        writer.add_scalar('train_main_loss', train_main_loss.avg, curr_iter)
        writer.add_scalar('train_aux_loss', train_aux_loss.avg, curr_iter)
        writer.add_scalar('lr', optimizer.param_groups[1]['lr'], curr_iter)

        if (i + 1) % train_args['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [train main loss %.5f], [train aux loss %.5f]. [lr %.10f]' % (
                epoch, i + 1, len(train_loader), train_main_loss.avg, train_aux_loss.avg,
                optimizer.param_groups[1]['lr']))

def validate(val_loader, net, criterion, optimizer, epoch, train_args, restore, visualize):
    # only validate when batch_size==1
    net.eval()
    val_loss = AverageMeter()

    gts_all, predictions_all = [], []

    for vi, data in enumerate(val_loader):
        input, gt, slices_info = data
        assert len(input.size()) == 5 and len(gt.size()) == 4 and len(slices_info.size()) == 3
        input.transpose_(0, 1)
        gt.transpose_(0, 1)
        slices_info.squeeze_(0)
        assert input.size()[3:] == gt.size()[2:]

        img_h = slices_info[-1,0].item() + slices_info[-1,4].item()
        img_w = slices_info[-1,2].item() + slices_info[-1,5].item()
        count = torch.zeros(img_h, img_w).cuda()
        output = torch.zeros(train_args['num_classes'], img_h, img_w).cuda()
        img_gt = np.zeros((img_h, img_w))

        slice_batch_pixel_size = input.size(1) * input.size(3) * input.size(4)

        for input_slice, gt_slice, info in zip(input, gt, slices_info):
            input_slice = Variable(input_slice).cuda()
            gt_slice = Variable(gt_slice).cuda()

            output_slice = net(input_slice)
            assert output_slice.size()[2:] == gt_slice.size()[1:]
            assert output_slice.size()[1] == train_args['num_classes']

            output[:, info[0]: info[0]+info[4], info[2]: info[2]+info[5]] += output_slice[0, :, :info[4], :info[5]].data
            count[info[0]: info[0]+info[4], info[2]: info[2]+info[5]] += 1
            img_gt[info[0]: info[0]+info[4], info[2]: info[2]+info[5]] += gt_slice[0, :info[4], :info[5]].data.cpu().numpy()

            val_loss.update(criterion(output_slice, gt_slice).data[0], slice_batch_pixel_size)

        output /= count
        prediction = output.max(0)[1].squeeze_(0).cpu().numpy()
        gts_all.append(img_gt/count.cpu().numpy().astype(int))
        predictions_all.append(prediction)

        print('validating: %d / %d' % (vi + 1, len(val_loader)))

        # visualize output result
        # if train_args['val_save_to_img_file']:
        #     for i in range(predictions.shape[0]):
        #         weighted_img = overlay_mask(restore(inputs[i].cpu()), predictions[i])
        #         cv2.imwrite('./vis/'+str(vi*predictions.shape[0]+i)+'.png', weighted_img)

        #         pred_pil = colorize_mask(predictions[i])
        #         pred_pil.save('./pred/'+str(vi*predictions.shape[0]+i)+'.png')
        #         gt_pil = colorize_mask(gts.data.cpu().numpy()[i])
        #         gt_pil.save('./gts/'+str(vi*predictions.shape[0]+i)+'.png')

    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, train_args['num_classes'])

    if mean_iu > train_args['best_record']['mean_iu']:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc
        snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
            epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[1]['lr']
        )
        torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_' + snapshot_name + '.pth'))

    print('--------------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
        train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
        train_args['best_record']['mean_iu'], train_args['best_record']['fwavacc'], train_args['best_record']['epoch']))

    print('--------------------------------------------------------------------')

    writer.add_scalar('val_loss', val_loss.avg, epoch)
    writer.add_scalar('acc', acc, epoch)
    writer.add_scalar('acc_cls', acc_cls, epoch)
    writer.add_scalar('mean_iu', mean_iu, epoch)
    writer.add_scalar('fwavacc', fwavacc, epoch)
    writer.add_scalar('lr', optimizer.param_groups[1]['lr'], epoch)

    net.train()

def overlay_mask(pilimage, pred):
    img = np.array(pilimage)
    weighted_img = None
    for i in range(train_args['num_classes']):
        points = np.where(pred==i)
        points = np.concatenate([points[1][:, np.newaxis], points[0][:, np.newaxis]], 1)
        if points.shape[0]>0:
            mask = np.ones((img.shape[0], img.shape[1], 3))*255
            cv2.fillPoly(mask, [points], (palette[3*i+2], palette[3*i+1], palette[3*i]))
            if weighted_img is None:
                weighted_img = cv2.addWeighted(img, 0.5, mask.astype('uint8'), 0.5, 0)
            else:
                weighted_img = cv2.addWeighted(weighted_img, 0.5, mask.astype('uint8'), 0.5, 0)
    return weighted_img

def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

if __name__ == '__main__':
    main(train_args)
