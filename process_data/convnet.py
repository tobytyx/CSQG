import os
import torch as t
from utils.util import get_free_gpu
USE_CUDA = t.cuda.is_available()
device = t.device("cuda" if USE_CUDA else "cpu")
if USE_CUDA:
    gpu_id = get_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
import torch.nn as nn
import copy
import torchvision.models as models


resnet_names = sorted(name for name in models.__dict__
    if name.islower()
    and name.startswith("resnet")
    and callable(models.__dict__[name]))

vgg_names = sorted(name for name in models.__dict__
    if name.islower()
    and name.startswith("vgg")
    and callable(models.__dict__[name]))

def factory(opt, cuda=True, data_parallel=True):
    opt = copy.deepcopy(opt)
    if not isinstance(opt, dict):
        opt = vars(opt)

    class WrapperModule(nn.Module):
        def __init__(self, net, forward_fn):
            super(WrapperModule, self).__init__()
            self.net = net
            self.forward_fn = forward_fn

        def forward(self, x):
            return self.forward_fn(self.net, x)

        def __getattr__(self, attr):
            try:
                return super(WrapperModule, self).__getattr__(attr)
            except AttributeError:
                return getattr(self.net, attr)

    def forward_resnet(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if opt["feature"] == 'layer4':
            return x
        x = self.avgpool(x)
        if opt["feature"] == 'avgpool':
            return x.squeeze()
        x = self.fc(x)
        if opt["feature"] == 'fc':
            return x.squeeze()

    def forward_vgg(self, x):
        features = list(self.features)
        classfier = list(self.classifier)
        features = t.nn.ModuleList(features).eval()
        classfier = t.nn.ModuleList(classfier).eval()
        result = {}
        for i, layer in enumerate(features):
            x = layer(x)
        result["conv5_3"] = x
        x = x.view(x.shape[0], -1)  # 将x转为[batch_size, 25088]
        for i, layer in enumerate(classfier):
            x = layer(x)
            if i == 3:
                result['fc7'] = x
            elif i == 6:
                result['fc8'] = x
        if opt['feature'] == 'fc7':
            return result['fc7']
        elif opt['feature'] == 'fc8':
            return result['fc8']
        elif opt['feature'] == 'conv5_3':
            return result["conv5_3"]
        else:
            raise ValueError

    model = models.__dict__[opt['arch']](pretrained=True)
    if opt['arch'] in resnet_names:
        model = WrapperModule(model, forward_resnet)  # ugly hack in case of DataParallel wrapping
    elif opt['arch'] in vgg_names:
        model = WrapperModule(model, forward_vgg)

    if data_parallel:
        model = nn.DataParallel(model).to(device)
        if not cuda:
            raise ValueError

    return model