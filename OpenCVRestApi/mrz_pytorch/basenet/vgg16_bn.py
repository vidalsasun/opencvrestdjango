from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
from torchvision.models.vgg import model_urls

import logging
logger = logging.getLogger('django.server')

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class vgg16_bn(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=True):

        logger.error('11111')

        super(vgg16_bn, self).__init__()

        logger.error('22222')

        #model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')
        model_urls['vgg16_bn'] = model_urls['vgg16_bn']
        
        logger.error('33333')

        vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features

        logger.error('44444')

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        logger.error('55555')

        for x in range(12):         # conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):         # conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):         # conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):         # conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        logger.error('66666')

        # fc6, fc7 without atrous conv
        self.slice5 = torch.nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.Conv2d(1024, 1024, kernel_size=1)
        )

        logger.error('77777')

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        
        logger.error('88888')

        init_weights(self.slice5.modules())        # no pretrained model for fc6 and fc7

        logger.error('99999')

        if freeze:
            for param in self.slice1.parameters():      # only first conv
                param.requires_grad= False

        logger.error('1010101010')

    def forward(self, X):
        logger.error('f1')
        h = self.slice1(X)
        logger.error('f2')
        h_relu2_2 = h
        logger.error('f3')
        h = self.slice2(h)
        logger.error('f4')
        h_relu3_2 = h
        logger.error('f5')
        h = self.slice3(h)
        logger.error('f6')
        h_relu4_3 = h
        logger.error('1010101010')
        h = self.slice4(h)
        logger.error('f7')
        h_relu5_3 = h
        logger.error('f8')
        h = self.slice5(h)
        logger.error('f9')
        h_fc7 = h
        logger.error('f10')
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        logger.error('f11')
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        logger.error('f12')
        return out
