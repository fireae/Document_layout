import os
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

# here (https://github.com/pytorch/vision/tree/master/torchvision/models) to find the download link of pretrained models
model_urls = {
    'resnet101':            'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152':            'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'inception_v3_google':  'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'vgg19_bn':             'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'vgg16':                'https://download.pytorch.org/models/vgg16-397923af.pth',
    'densenet201':          'https://download.pytorch.org/models/densenet201-c1103571.pth',
}


root = '/data/weihong_ma/experiment/exp1_pytorch-semantic-segmentation/pretrained'
res101_path = os.path.join(root, 'ResNet', 'resnet101-5d3b4d8f.pth')
res152_path = os.path.join(root, 'ResNet', 'resnet152-b121ed2d.pth')
inception_v3_path = os.path.join(root, 'Inception', 'inception_v3_google-1a9a5a14.pth')
vgg19_bn_path = os.path.join(root, 'VggNet', 'vgg19_bn-c79401a0.pth')
vgg16_path = os.path.join(root, 'VggNet', 'vgg16-397923af.pth')
dense201_path = os.path.join(root, 'DenseNet', 'densenet201-4c113574.pth')

path_list = [res101_path, res152_path, inception_v3_path, vgg19_bn_path, vgg16_path, dense201_path]
for i in path_list:
    if not os.path.exists(i):
        name = i.split('/')[-1].split('-')[0]
        save_dir = i.replace(i.split('/')[-1], "")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        download_url(model_urls[name], i)



'''
vgg16 trained using caffe
visit this (https://github.com/jcjohnson/pytorch-vgg) to download the converted vgg16
'''
vgg16_caffe_path = os.path.join(root, 'VggNet', 'vgg16-caffe.pth')
