import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import pandas as pd


class AFSM(nn.Module):
    """
    Feature Mutual Reconstruction Module
    """
    def __init__(self, num_channel_tran=768,num_channel=640):
        super(AFSM, self).__init__()

        self.cov1 = nn.Conv2d(num_channel_tran, num_channel, kernel_size=1, stride=1)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.maxpool2 = nn.MaxPool2d(3, stride=1, padding=0)
        self.w3 = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.w4 = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.w5 = nn.Parameter(torch.FloatTensor([0.1]), requires_grad=True)




    def forward(self, features_a, features_b, cls_features):

        batch_size = features_a.size(0)
        b, c_t, hw = features_b.size()
        features_b = features_b.permute(0, 2, 1).view(batch_size, hw, 14, 14)
        features_b = self.maxpool1(features_b)
        features_b = self.maxpool2(features_b)
        cls_features = cls_features.unsqueeze(-1).unsqueeze(-1)
        features_b = self.cov1(features_b)
        cls_features = self.cov1(cls_features)
        cls_features = self.w5 * cls_features

        # if self.resnet:
        #     feature_map = feature_map/np.sqrt(640)
        # feature_map_avg = self.avgpool(feature_map)
        features_avg = self.maxpool(features_a)
        z_cls_feature = torch.cat((features_avg, cls_features), dim=1)
        z_cls_feature = z_cls_feature.view(z_cls_feature.size(0), z_cls_feature.size(1))
        w_count = F.softmax(z_cls_feature, dim=1)
        sum1 = w_count[:, :640].sum(dim=1)
        sum2 = w_count[:, 640:].sum(dim=1)
        feature_map = self.w3 * features_a * sum1.view(w_count.size(0), 1, 1, 1)
        feature_map_tran = self.w4 * features_b * sum2.view(w_count.size(0), 1, 1, 1)



        return feature_map, feature_map_tran