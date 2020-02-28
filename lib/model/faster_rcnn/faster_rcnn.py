# --------------------------------------------------------
# Pytorch Meta R-CNN
# Written by Anny Xu, Xiaopeng Yan, based on the code from Jianwei Yang
# --------------------------------------------------------
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
import pickle
import math

import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import os

class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, num_ways, class_agnostic, meta_train, meta_test=None, meta_loss=None, transductive=None, visualization=None):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.meta_train = meta_train
        self.meta_test = meta_test
        self.meta_loss = meta_loss
        self.simloss = True
        self.dis_simloss = True
        self.transductive = transductive
        self.visualization = visualization

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()
        self.num_layers_g = 3
        self.num_ways = num_ways
        self.alpha = 0.5


    def forward(self, im_data_list, im_info_list, gt_boxes_list, num_boxes_list, average_shot=None,
                mean_class_attentions=None):
        # return attentions for testing
        if average_shot:
            prn_data = im_data_list[0]  # len(metaclass)*4*224*224
            attentions, attentions_t = self.prn_network(prn_data)
            return attentions, attentions_t
        # extract attentions for training
        if self.meta_train and self.training:
            prn_data = im_data_list[0]  # len(metaclass)*4*224*224
            # feed prn data to prn_network
            attentions, attentions_t = self.prn_network(prn_data)
            prn_cls = im_info_list[0]  # len(metaclass)

        im_data = im_data_list[-1]
        im_info = im_info_list[-1]
        gt_boxes = gt_boxes_list[-1]
        num_boxes = num_boxes_list[-1]

        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(self.rcnn_conv1(im_data))

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phase, then use ground truth bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)

        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'crop':
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))  # (b*128)*1024*7*7
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)  # (b*128)*2048

        # meta training phase
        if self.meta_train:
            rcnn_loss_bbox = []
            support_data = attentions_t
            support_label = torch.cat(prn_cls, dim=0) + torch.ones(self.num_ways).long()
            support_data = support_data.unsqueeze(0).repeat(batch_size, 1, 1)
            support_label = support_label.unsqueeze(0).repeat(batch_size, 1).cuda()
            query_label = rois_label.view(batch_size, -1)
            query_data = pooled_feat.contiguous().view(batch_size, cfg.TRAIN.BATCH_SIZE, -1)
           # print(support_data, query_data)
            num_supports = support_data.size(1)
            num_queries = query_data.size(1)
            num_samples = num_supports + num_queries
            support_edge_mask = Variable(torch.zeros(batch_size, num_samples, num_samples)).cuda()
            support_edge_mask[:, :num_supports, :num_supports] = 1
            query_edge_mask = 1 - support_edge_mask
            full_data = torch.cat((support_data, query_data), 1)
            full_label = torch.cat((support_label, query_label.data), 1)
            full_edge = self.label2edge(full_label)

            if self.visualization:
                edge_png = full_edge[0, 0, :, :].detach().data.cpu()
                edge_png = edge_png.item() if edge_png.dim() == 0 else edge_png.numpy()
                ax = sns.heatmap(edge_png, xticklabels=False, yticklabels=False, linewidth=0.1, cmap="coolwarm",
                                 cbar=False, square=True)
                ax.get_figure().savefig('visualization/gt.png')

            node_png = full_data[0, :, :].detach().data.cpu()
            node_png = node_png.item() if node_png.dim() == 0 else node_png.numpy()

            '''os.remove('visualization/node.png')
            ax = sns.heatmap(node_png, cmap='rainbow')
            ax.get_figure().savefig('visualization/node.png')
            plt.close()'''

            init_edge = full_edge.clone()
            init_edge[:, :, num_supports:, :] = 0.5
            init_edge[:, :, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, 0, num_supports + i, num_supports + i] = 1.0
                init_edge[:, 1, num_supports + i, num_supports + i] = 0.0

            '''full_data = F.normalize(full_data, p=2, dim=-1)
            try_edge = torch.zeros(143, 143)
            for i in range(143):
                for j in range(143):
                    try_edge[i, j] = torch.dist(full_data.data[0, i, :], full_data.data[0, j, :], p = 2)'''

            #print(full_edge[0, 0, :20, :20],try_edge[:20, :20])
            #print(full_edge[0, 0, 15:, 15:],try_edge[15:, 15:])
            #print(full_edge[0, 0, :],try_edge)

            if self.transductive:
                full_logit_layers = self.RCNN_cls_score(batch_size, full_data, init_edge, num_supports, num_queries, full_label)
            else:
                support_data_tiled = support_data.unsqueeze(1).repeat(1, num_queries, 1, 1)
                support_data_tiled = support_data_tiled.view(batch_size * num_queries, num_supports, -1)
                query_data_reshaped = query_data.contiguous().view(batch_size * num_queries, -1).unsqueeze(1)
                input_node_feat = torch.cat([support_data_tiled, query_data_reshaped], 1)
                input_edge_feat = Variable(0.5 * torch.ones(batch_size, 2, num_supports + 1, num_supports + 1)).cuda()

                input_edge_feat[:, :, :num_supports, :num_supports] = init_edge[:, :, :num_supports, :num_supports]
                input_edge_feat = input_edge_feat.repeat(num_queries, 1, 1, 1)

                full_logit_layers = self.RCNN_cls_score(batch_size, input_node_feat, input_edge_feat, num_supports, num_queries, full_label)

            query_node_pred_layers = [torch.bmm(full_logit_layer[:, 0, num_supports:, :num_supports],
                                                self.one_hot_encode(self.n_classes, support_label)) for
                                      full_logit_layer
                                      in full_logit_layers]
            cls_score = query_node_pred_layers[-1]
            cls_score = cls_score.view(batch_size * num_queries, -1)
            cls_score_m = cls_score.clone()
            for i in range(batch_size * num_queries):
                cls_score_m[i, 0] = 1 - torch.max(cls_score[i, :])

            if self.training:
                # classification loss
                full_edge_loss_layers = [- (
                        (1 - full_edge[:, 0, :, :]) * torch.log(1 - full_logit_layer[:, 0, :, :]) + (
                        1 - (1 - full_edge[:, 0, :, :])) * torch.log(
                    1 - (1 - full_logit_layer[:, 0, :, :]))) for full_logit_layer in full_logit_layers]

                if not self.transductive:
                    for l in range(3):
                        full_edge_loss_layers[l][:, num_supports:, num_supports:] = 0

                pos_query_edge_loss_layers = [
                    torch.sum(full_edge_loss_layer * query_edge_mask * full_edge[:, 0]) / torch.sum(
                        query_edge_mask * full_edge[:, 0]) for full_edge_loss_layer in full_edge_loss_layers]

                neg_query_edge_loss_layers = [
                    torch.sum(full_edge_loss_layer * query_edge_mask * (1 - full_edge[:, 0])) / torch.sum(
                       query_edge_mask * (1 - full_edge[:, 0])) for full_edge_loss_layer in
                    full_edge_loss_layers]
                query_edge_loss_layers = [pos_query_edge_loss_layer + neg_query_edge_loss_layer for
                                          (pos_query_edge_loss_layer, neg_query_edge_loss_layer) in
                                          zip(pos_query_edge_loss_layers, neg_query_edge_loss_layers)]
                total_loss_layers = query_edge_loss_layers
                #test_loss = []
                #for l in range(self.num_layers_g - 1):
                #    test_loss += [pos_query_edge_loss_layers[l].view(-1) * 0.5]
                #rcnn_loss_cls = torch.mean(torch.cat(test_loss, 0))

                total_loss = []
                for l in range(self.num_layers_g - 1):
                    total_loss += [total_loss_layers[l] * 0.5]
                total_loss += [total_loss_layers[-1] * 1.0]
                rcnn_loss_cls = torch.mean(torch.cat(total_loss, 0))
                #rcnn_loss_cls = F.cross_entropy(cls_score_m, rois_label)

            # pooled feature maps need to operate channel-wise multiplication with the corresponding class's attentions of every roi of image
            for b in range(batch_size):
                zero = Variable(torch.FloatTensor([0]).cuda())
                proposal_labels = rois_label[b * 128:(b + 1) * 128].data.cpu().numpy()[0]
                unique_labels = list(np.unique(proposal_labels)) # the unique rois labels of the input image

                for i in range(attentions.size(0)):  # attentions len(attentions)*2048
                    if prn_cls[i].numpy()[0] + 1 not in unique_labels:
                        rcnn_loss_bbox.append(zero)
                        continue

                    channel_wise_feat = pooled_feat[b * cfg.TRAIN.BATCH_SIZE:(b + 1) * cfg.TRAIN.BATCH_SIZE, :] * \
                                        attentions[i]  # 128x2048 channel-wise multiple

                    bbox_pred = self.RCNN_bbox_pred(channel_wise_feat)  # 128 * 4
                    if self.training and not self.class_agnostic:
                        # select the corresponding columns according to roi labels
                        bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                        bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                                        rois_label[
                                                        b * cfg.TRAIN.BATCH_SIZE:(b + 1) * cfg.TRAIN.BATCH_SIZE].view(
                                                            rois_label[b * cfg.TRAIN.BATCH_SIZE:(
                                                                                                        b + 1) * cfg.TRAIN.BATCH_SIZE].size(
                                                                0), 1, 1).expand(
                                                            rois_label[b * cfg.TRAIN.BATCH_SIZE:(
                                                                                                        b + 1) * cfg.TRAIN.BATCH_SIZE].size(
                                                                0), 1,
                                                            4))
                        bbox_pred = bbox_pred_select.squeeze(1)


                        RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target[b * 128:(b + 1) * 128],
                                                         rois_inside_ws[b * 128:(b + 1) * 128],
                                                         rois_outside_ws[b * 128:(b + 1) * 128])

                        rcnn_loss_bbox.append(RCNN_loss_bbox)
            # meta attentions loss
            if self.meta_loss:
                attentions_score = self.Meta_cls_score(attentions)
                meta_loss = F.cross_entropy(attentions_score, Variable(support_label[0]))
            else:
                meta_loss = 0
            # simloss
            if self.simloss:
                mask_p = full_edge[:, 0, num_supports:, :].clone()
                node_sim = full_logit_layers[-1][:, 0, num_supports:, :] * mask_p
                node_sim_c = (torch.sum(node_sim, -1) / torch.sum(mask_p, -1)).unsqueeze(-1).repeat(1, 1, num_samples)
                norm = (node_sim - node_sim_c).norm(p=2, dim=-1)
                param = (1 / torch.sum(mask_p, -1)) * 1/2
                loss = norm * norm * param
                SIMLOSS = loss.mean()

            # dis_simloss
            if self.dis_simloss:
                dis_simloss = []
                mask_p = full_edge[:, 0, num_supports:, :].clone()
                mask_p[:, :, num_supports:] = 0
                mask_n = 1 - full_edge[:, 0, num_supports:, :].clone()
                for b in range(batch_size):
                    for i in range(num_queries):
                        if torch.sum(mask_p[b, i, :]).data[0] == 0:
                            node_sim = full_logit_layers[-1][b, 0, num_supports+i, :] * full_edge[b, 0, num_supports+i, :]
                        else:
                            node_sim = full_logit_layers[-1][b, 0, num_supports+i:, :] * mask_p[b, i, :]
                        other_sim = full_logit_layers[-1][b, 0, num_supports+i:, :] * mask_n[b, i, :]
                        max_sim = torch.max(node_sim)
                        max_other = torch.max(other_sim)
                        loss = F.relu(max_other - max_sim + self.alpha)
                        dis_simloss.append(loss)
                DIS_SIMLOSS = sum(dis_simloss) / len(dis_simloss)


            return rois, rpn_loss_cls, rpn_loss_bbox, rcnn_loss_cls, rcnn_loss_bbox, rois_label, 0, 0, meta_loss, SIMLOSS, DIS_SIMLOSS

        elif self.meta_test:
            cls_prob_list = []
            bbox_pred_list = []
            support_data_list = []
            support_label_list = []
            mean_class_attentions_t = pickle.load(open(os.path.join('attentions',str(2) + '_shots_' + str(10) + '_mean_class_attentions_t.pkl'), 'rb'))

            for key, value in mean_class_attentions_t.items():
                support_data_list.append(value)
                support_label_list.append(key)
            support_data = torch.stack(support_data_list, 0).unsqueeze(0).repeat(batch_size, 1, 1)
            support_label = torch.Tensor(support_label_list).long() + torch.ones(self.num_ways).long()
            support_label = support_label.unsqueeze(0).repeat(batch_size, 1).cuda()
            query_data = pooled_feat.contiguous().view(batch_size, -1, 2048)
            num_queries = query_data.size(1)
            num_supports = len(mean_class_attentions_t)
            num_samples = num_supports + num_queries
            full_data = torch.cat((support_data, query_data), 1)
            init_edge_0 = torch.zeros(batch_size, 1, num_samples, num_samples)
            init_edge_1 = torch.ones(batch_size, 1,  num_samples, num_samples)
            init_edge = Variable(torch.cat([init_edge_0, init_edge_1], 1)).cuda()

            init_edge[:, :, num_supports:, :] = 0.5
            init_edge[:, :, :, num_supports:] = 0.5
            for i in range(num_samples):
                init_edge[:, 0, i, i] = 1.0
                init_edge[:, 1, i, i] = 0.0

            if self.training:
                query_label = rois_label.view(batch_size, -1)
                full_label = torch.cat((support_label, query_label.data), 1)
                full_edge = self.label2edge(full_label)
                support_edge_mask = Variable(torch.zeros(batch_size, num_samples, num_samples)).cuda()
                support_edge_mask[:, :num_supports, :num_supports] = 1
                query_edge_mask = 1 - support_edge_mask

            if self.transductive:
                full_logit_layers = self.RCNN_cls_score(batch_size, full_data, init_edge, num_supports, num_queries)
            else:
                support_data_tiled = support_data.unsqueeze(1).repeat(1, num_queries, 1, 1)
                support_data_tiled = support_data_tiled.view(batch_size * num_queries, num_supports, -1)
                query_data_reshaped = query_data.contiguous().view(batch_size * num_queries, -1).unsqueeze(1)
                input_node_feat = torch.cat([support_data_tiled, query_data_reshaped], 1)
                input_edge_feat = Variable(0.5 * torch.ones(batch_size, 2, num_supports + 1, num_supports + 1)).cuda()

                input_edge_feat[:, :, :num_supports, :num_supports] = init_edge[:, :, :num_supports, :num_supports]
                input_edge_feat = input_edge_feat.repeat(num_queries, 1, 1, 1)

                full_logit_layers = self.RCNN_cls_score(batch_size, input_node_feat, input_edge_feat, num_supports, num_queries)

            # compute object classification probability
            query_node_pred_layers = [torch.bmm(full_logit_layer[:, 0, num_supports:, :num_supports],
                                                self.one_hot_encode(self.n_classes, support_label)) for
                                      full_logit_layer
                                      in full_logit_layers]
            cls_score_m = query_node_pred_layers[-1]
            cls_score_m = cls_score_m.view(batch_size * num_queries, -1)
            cls_score = cls_score_m.clone()
            for i in range(batch_size * num_queries):
                cls_score[i, 0] = 1 - torch.max(cls_score_m[i, :])
            print(cls_score)
            cls_prob = F.softmax(cls_score)
            print(cls_prob)

            '''if self.training:
                # classification loss
               full_edge_loss_layers = [- (
                        (1 - full_edge[:, 0, :, :]) * torch.log(1 - full_logit_layer[:, 0, :, :]) + (
                        1 - (1 - full_edge[:, 0, :, :])) * torch.log(
                    1 - (1 - full_logit_layer[:, 0, :, :]))) for full_logit_layer in full_logit_layers]

               if not self.transductive:
                    for l in range(3):
                        full_edge_loss_layers[l][:, num_supports:, num_supports:] = 0

                pos_query_edge_loss_layers = [
                    torch.sum(full_edge_loss_layer * query_edge_mask * full_edge[:, 0]) / torch.sum(
                        query_edge_mask * full_edge[:, 0]) for full_edge_loss_layer in full_edge_loss_layers]

                neg_query_edge_loss_layers = [
                    torch.sum(full_edge_loss_layer * query_edge_mask * (1 - full_edge[:, 0])) / torch.sum(
                       query_edge_mask * (1 - full_edge[:, 0])) for full_edge_loss_layer in
                    full_edge_loss_layers]
                query_edge_loss_layers = [pos_query_edge_loss_layer + neg_query_edge_loss_layer for
                                          (pos_query_edge_loss_layer, neg_query_edge_loss_layer) in
                                          zip(pos_query_edge_loss_layers, neg_query_edge_loss_layers)]
                total_loss_layers = query_edge_loss_layers
                #test_loss = []
                #for l in range(self.num_layers_g - 1):
                #    test_loss += [pos_query_edge_loss_layers[l].view(-1) * 0.5]
                #rcnn_loss_cls = torch.mean(torch.cat(test_loss, 0))

                total_loss = []
                for l in range(self.num_layers_g - 1):
                    total_loss += [total_loss_layers[l] * 0.5]
                total_loss += [total_loss_layers[-1] * 1.0]
                RCNN_loss_cls = torch.mean(torch.cat(total_loss, 0))'''

            for i in range(num_supports):
                mean_attentions = mean_class_attentions[i]
                channel_wise_feat = pooled_feat * mean_attentions
                # compute bbox offset
                bbox_pred = self.RCNN_bbox_pred(channel_wise_feat)
                if self.training and not self.class_agnostic:
                    # select the corresponding columns according to roi labels
                    bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                    bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                                    rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0),
                                                                                                     1, 4))
                    bbox_pred = bbox_pred_select.squeeze(1)

                RCNN_loss_bbox = 0
                RCNN_loss_cls = 0

                if self.training:
                    # bounding box regression L1 loss
                    RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

                cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
                bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
                cls_prob_list.append(cls_prob)
                bbox_pred_list.append(bbox_pred)

            return rois, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, cls_prob_list, bbox_pred_list, 0, 0, 0
        else:
            bbox_pred = self.RCNN_bbox_pred(pooled_feat)
            if self.training and not self.class_agnostic:
                # select the corresponding columns according to roi labels
                bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                                rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1,
                                                                                                 4))
                bbox_pred = bbox_pred_select.squeeze(1)

            # compute object classification probability
            cls_score = self.RCNN_cls_score_n(pooled_feat)  # 128 * 1001
            cls_prob = F.softmax(cls_score)

            RCNN_loss_cls = 0
            RCNN_loss_bbox = 0

            if self.training:
                # classification loss
                RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

                # bounding box regression L1 loss
                RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

            cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
            bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, cls_prob, bbox_pred, 0, 0, 0

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        def uniform_init(m, stdv):
            m.weight.data.uniform_(-stdv, stdv)
            m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score_n, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def label2edge(self, label):
        # get size
        num_samples = label.size(1)
        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)
        # compute edge
        edge = torch.eq(label_i, label_j).float()
        # expand
        edge = edge.unsqueeze(1)
        edge = torch.cat((edge, 1 - edge), 1)
        return Variable(edge)
