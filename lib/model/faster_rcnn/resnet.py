# --------------------------------------------------------
# Pytorch Meta R-CNN
# Written by Anny Xu, Xiaopeng Yan, based on code from Jianwei Yang
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
import pdb
from collections import OrderedDict
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)

def init_conv(conv,glu=True):
  init.xavier_uniform(conv.weight)
  if conv.bias is not None:
    conv.bias.data.zero_()

def init_linear(linear):
  init.constant(linear.weight,0)
  init.constant(linear.bias, 1)

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    # block = Bottleneck, planes = 512, blocks = 3, stride = 2
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet18(pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model


def resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model

class NodeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 1],
                 dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):

            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features * 3,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)


        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):

        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)

        diag_mask = Variable(1.0 - torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 2, 1, 1)).cuda()
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)
        aggr_feat = torch.bmm(torch.cat(torch.split(edge_feat, 1, 1), 2).squeeze(1), node_feat)
        node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(num_data, 1), -1)], -1).transpose(1, 2)
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)


        return node_feat



class EdgeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 2, 1, 1],
                 separate_dissimilarity=False,
                 dropout=0.0):
        super(EdgeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                       out_channels=self.num_features_list[l],
                                                       kernel_size=1,
                                                       bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                           out_channels=1,
                                           kernel_size=1,
                                           bias=False)
        layer_list['norm_last'] = nn.BatchNorm2d(num_features=1,)
        self.sim_network = nn.Sequential(layer_list)

        if self.separate_dissimilarity:
            # layers
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                # set layer
                layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                           out_channels=self.num_features_list[l],
                                                           kernel_size=1,
                                                           bias=False)
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                                )
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()

                if self.dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=self.dropout)

            layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                               out_channels=1,
                                               kernel_size=1,
                                               bias=False)
            layer_list['norm_last'] = nn.BatchNorm2d(num_features=1, )
            self.dsim_network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # compute abs(x_i, x_j)
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)

        #print(self.sim_network(x_ij)[0])
        sim_val = F.sigmoid(self.sim_network(x_ij))
        #print(sim_val[0])
        if self.separate_dissimilarity:
           dsim_val = F.sigmoid(self.dsim_network(x_ij))
        else:
           dsim_val = 1.0 - sim_val

        diag_mask = Variable(1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 2, 1, 1).cuda())
        edge_feat = edge_feat * diag_mask
        merge_sum = torch.sum(edge_feat, -1, True)
        edge_feat = F.normalize(torch.cat([sim_val, dsim_val], 1) * edge_feat, p=1, dim=-1) * merge_sum
        force_edge_feat = torch.cat((torch.eye(node_feat.size(1)).unsqueeze(0), torch.zeros(node_feat.size(1), node_feat.size(1)).unsqueeze(0)), 0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).cuda()
        edge_feat = edge_feat + Variable(force_edge_feat)
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1).repeat(1, 2, 1, 1)

        return edge_feat


class GraphNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 node_features,
                 edge_features,
                 num_layers,
                 visualization,
                 dropout=0.0):
        super(GraphNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout
        self.visualization = visualization

        # for each layer
        node_init = nn.Sequential(OrderedDict([('linear', nn.Linear(2048, 128, bias=True)), ('norm', nn.BatchNorm1d(128))]))
        #node_init = nn.Sequential(OrderedDict([('linear', nn.Linear(2048, 128, bias=True))]))
        self.add_module('node_init', node_init)
        for l in range(self.num_layers):
            # set edge to node
            edge2node_net = NodeUpdateNetwork(in_features=self.in_features if l == 0 else self.node_features,
                                              num_features=self.node_features,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            # set node to edge
            node2edge_net = EdgeUpdateNetwork(in_features=self.node_features,
                                              num_features=self.edge_features,
                                              separate_dissimilarity=True,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            self.add_module('edge2node_net{}'.format(l), edge2node_net)
            self.add_module('node2edge_net{}'.format(l), node2edge_net)

    # forward
    def forward(self, node_feat, edge_feat, full_label):
        num_t = node_feat.size(0)
        try_edge = torch.zeros(143, 143)
        # for each layer
        edge_feat_list = []
        node_feat_list = []
        #node_feat = self._modules['node_init'](node_feat.view(-1, 2048)).view(num_t, -1, 128)
        node_feat_0 = self._modules['node_init'](node_feat[:, :21, :].contiguous().view(-1, 2048)).contiguous().view(num_t, -1, 128)
        node_feat_1 = self._modules['node_init'](node_feat[:, 21:, :].contiguous().view(-1, 2048)).contiguous().view(num_t, -1, 128)
        node_feat = torch.cat([node_feat_0, node_feat_1], 1)
        '''for i in range(143):
            for j in range(143):
                try_edge[i, j] = torch.dist(node_feat.data[0, i, :], node_feat.data[0, j, :])
        print(try_edge[15:, 15:])'''
        #node_feat = F.normalize(node_feat, dim=2)
        # normalize还是batchnorm？？normalize是否可以，会不会让background和foreground的区别降低，怎么样才能让support和query的特征保持一致？
        node_feat_list.append(node_feat)
        for l in range(self.num_layers):
            # (1) edge to node
            # print(node_feat[0])
            node_feat = self._modules['edge2node_net{}'.format(l)](node_feat, edge_feat)

            # (2) node to edge
            edge_feat = self._modules['node2edge_net{}'.format(l)](node_feat, edge_feat)

            # save edge feature
            edge_feat_list.append(edge_feat)
            node_feat_list.append(node_feat)

        if self.visualization:
            for l in range(self.num_layers):
                edge_png = edge_feat_list[l][0, 0, :, :].detach().data.cpu()
                edge_png = edge_png.item() if edge_png.dim() == 0 else edge_png.numpy()
                ax = sns.heatmap(edge_png, xticklabels=False, yticklabels=False, linewidth=0.1,  cmap="coolwarm",  cbar=False, square=True)
                ax.get_figure().savefig('visualization/edge_feat_layer{}.png'.format(l))

            for l in range(self.num_layers + 1):
                node_png = node_feat_list[l][0, :, :].detach().data.cpu().numpy()
                node_label = full_label.cpu().numpy()
                tsne = TSNE(perplexity=100, n_components=2, init='pca', random_state=501)
                node_tsne = tsne.fit_transform(node_png)
                x_min, x_max = node_tsne.min(0), node_tsne.max(0)
                X_norm = (node_tsne - x_min) / (x_max - x_min)
                ax = plt.figure()
                #figsize=(8, 8)

                start = 0.0
                stop = 1.0
                number_of_lines = 16
                cm_subsection = np.linspace(start, stop, number_of_lines)
                colors = plt.cm.Spectral(cm_subsection)

                for i in range(15):
                    plt.text(X_norm[i, 0], X_norm[i, 1], '*' + str(node_label[0][i]), color=colors[node_label[0][i]],
                             fontdict={'weight': 'bold', 'size': 18})
                for i in range(15, X_norm.shape[0]):
                    plt.text(X_norm[i, 0], X_norm[i, 1], 'o', color=colors[node_label[0][i]],
                             fontdict={'weight': 'bold', 'size': 18})
                ax.savefig('visualization/node_feat_layer{}.png'.format(l))
                plt.close(ax)


        return edge_feat_list

class resnet(_fasterRCNN):
  def __init__(self, classes, num_ways, num_layers=101, pretrained=False, class_agnostic=False, meta_train=True, meta_test=None, meta_loss=None, transductive=None, visualization=None):
    self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.meta_train = meta_train
    self.meta_test = meta_test
    self.transductive = transductive
    self.visualization = visualization


    _fasterRCNN.__init__(self, classes, num_ways, class_agnostic, meta_train, meta_test, meta_loss, transductive, visualization)

  def _init_modules(self):
    resnet = resnet101()

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    # Build resnet.
    self.meta_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.rcnn_conv1 = resnet.conv1

    self.RCNN_base = nn.Sequential(resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)


    self.RCNN_top = nn.Sequential(resnet.layer4)
    self.sigmoid = nn.Sigmoid()
    self.max_pooled = nn.MaxPool2d(2)

    self.RCNN_cls_score_n = nn.Linear(2048, self.n_classes)
    self.gnn_module = GraphNetwork(in_features=128,
                               node_features=96,
                               edge_features=96,
                               num_layers=self.num_layers_g,
                               visualization=self.visualization,
                               dropout=0.0)
    if self.meta_loss:
      self.Meta_cls_score = nn.Linear(2048, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 4) # x,y,w,h
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)


    # Fix blocks
    for p in self.rcnn_conv1.parameters(): p.requires_grad=False
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False


    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 5)
    if cfg.RESNET.FIXED_BLOCKS >= 4:
      for p in self.RCNN_top.parameters(): p.requires_grad = False
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base[4].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base[3].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_base[4].train()
      self.RCNN_base[5].train()

      self.RCNN_base.eval()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7

  def prn_network(self,im_data):
    '''
    the Predictor-head Remodeling Network (PRN)
    :param im_data:
    :return attention vectors:
    '''
    base_feat = self.RCNN_base(self.meta_conv1(im_data))
    feature = self._head_to_tail(self.max_pooled(base_feat))
    attentions = self.sigmoid(feature)
    # attentions_t = F.tanh(feature)
    return  attentions, feature

  def RCNN_cls_score(self, batch_size, full_data, init_edge, num_supports, num_queries, full_label=None):
      if self.transductive:
          full_logit_layers = self.gnn_module(node_feat=full_data, edge_feat=init_edge, full_label=full_label)

      else:
          num_samples = num_supports + num_queries
          logit_layers = self.gnn_module(node_feat=full_data, edge_feat=init_edge, full_label=full_label)
          logit_layers = [logit_layer.view(batch_size, num_queries, 2, num_supports + 1, num_supports + 1) for logit_layer in logit_layers]
          full_logit_layers = []
          for l in range(3):
              full_logit_layers.append(Variable(torch.zeros(batch_size, 2, num_samples, num_samples).cuda()))
          for l in range(3):
              full_logit_layers[l][:, :, :num_supports, :num_supports] = logit_layers[l][:, :, :, :num_supports, :num_supports].mean(1)
              full_logit_layers[l][:, :, :num_supports, num_supports:] = logit_layers[l][:, :, :, :num_supports, -1].transpose(1, 2).transpose(2, 3)
              full_logit_layers[l][:, :, num_supports:, :num_supports] = logit_layers[l][:, :, :, -1, :num_supports].transpose(1, 2)

      return full_logit_layers


  def one_hot_encode(self, num_classes, class_idx):
    batchsize = class_idx.size(0)
    class_idx = class_idx.view(-1).cpu()
    return Variable(torch.eye(num_classes)[class_idx].view(batchsize, -1, num_classes)).cuda()

