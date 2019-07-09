import datetime
import os
import random

import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import sys
sys.path.append("../../")
import utils.transforms as extended_transforms
import utils.joint_transforms as joint_transforms
from datasets import PRIMA
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d

cudnn.benchmark = False

ckpt_path = '../../ckpt'
exp_name = 'PRIMA-fcn8s'
writer = SummaryWriter(os.path.join(ckpt_path, 'exp', exp_name))

args = {
    'epoch_num': 300,
    'lr': 1e-10,
    'weight_decay': 1e-4,
    'momentum': 0.95,
    'lr_patience': 100,  # large patience denotes fixed lr
    'snapshot': '',  # empty string denotes learning from scratch
    'print_freq': 20,
    'val_save_to_img_file': False,
    'val_img_sample_rate': 0.1  # randomly sample some validation results to display
}

def main(train_args):
    net = FCN8s(num_classes=11)

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
        # joint_transforms.Scale(short_size),
        joint_transforms.Scale(2000),
        # joint_transforms.RandomCrop(args['input_size']),
        joint_transforms.RandomHorizontallyFlip()
    ])
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

    train_set = PRIMA('train', joint_transform=train_joint_transform, transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=1, num_workers=0, shuffle=True)

    it = iter(train_loader)
    first = next(it)
    import pdb; pdb.set_trace()

    criterion = CrossEntropyLoss2d(size_average=False).cuda()
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

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=train_args['lr_patience'], min_lr=1e-10, verbose=True)
    for epoch in range(curr_epoch, train_args['epoch_num'] + 1):
        train(train_loader, net, criterion, optimizer, epoch, train_args)
        # val_loss = validate(val_loader, net, criterion, optimizer, epoch, train_args, restore_transform, visualize)
        # scheduler.step(val_loss)



def train(train_loader, net, criterion, optimizer, epoch, train_args):
    train_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        assert inputs.size()[2:] == labels.size()[1:]
        N = inputs.size(0)
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        assert outputs.size()[2:] == labels.size()[1:]
        assert outputs.size()[1] == voc.num_classes

        loss = criterion(outputs, labels) / N
        loss.backward()
        optimizer.step()

        train_loss.update(loss.data[0], N)

        curr_iter += 1
        writer.add_scalar('train_loss', train_loss.avg, curr_iter)

        if (i + 1) % train_args['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg
            ))

def validate(val_loader, net, criterion, optimizer, epoch, train_args, restore, visualize):
    net.eval()

if __name__ == '__main__':
    main(args)