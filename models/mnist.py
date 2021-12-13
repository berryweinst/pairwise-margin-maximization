import torch
import torch.nn as nn
import os
import numpy as np
__all__ = ['mnist']

class mnist_model(nn.Module):

    def __init__(self, optimize_mms=False, dataset='mnist', depth='10'):
        super(mnist_model, self).__init__()
        self.optimize_mms = optimize_mms
        self.feats = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3,  1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3,  1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            # nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, 10)

        # self.classifier = nn.Conv2d(128, 10, 1)
        # self.avgpool = nn.AvgPool2d(6, 6)
        self.dropout = nn.Dropout(0.5)


    def mms_loss(self, logit, target, mode='mean', optimize=False):

        # logit_gt = torch.gather(logit, dim=1, index=target.unsqueeze(1).long()).squeeze()
        # logit_top1 = torch.zeros_like(logit_gt)
        # logit_top1_indices = torch.zeros_like(target)
        topk_vals, topk_indices = logit.topk(2, 1)
        # logit_top1 = torch.where(target.eq(topk_indices[:, 0]), topk_vals[:, 1], topk_vals[:, 0])
        logit_top1_indices = torch.where(target.eq(topk_indices[:, 0]), topk_indices[:, 1], topk_indices[:, 0])
        # for idx, (t, ii) in enumerate(zip(topk_vals, topk_indices)):
        #     if target[idx].long() != topk_indices[idx][0]:
        #         logit_top1[idx] = t[0]
        #         logit_top1_indices[idx] = ii[0]
        #     else:
        #         logit_top1[idx] = t[1]
        #         logit_top1_indices[idx] = ii[1]

        w1 = self.classifier.weight.index_select(0, target)
        w2 = self.classifier.weight.index_select(0, logit_top1_indices)

        # mms = (logit_gt - logit_top1) / (w1 - w2).norm(dim=-1)
        mms = (w1 - w2).norm(dim=-1)

        # topk_vals, pred_classes = logit.topk(2, 1)
        #
        # # with torch.no_grad():
        # w1 = self.fc.weight.index_select(0, pred_classes[:, 0])
        # w2 = self.fc.weight.index_select(0, pred_classes[:, 1])

        # mms = (topk_vals[:, 0] - topk_vals[:, 1]) / (w1 - w2).norm(dim=-1)


        if not optimize:
            mms = mms.detach()

        # mms = 1 / (1 + mms.mean())
        if mode == 'mean':
            mms = mms.mean()
        elif mode == 'min':
            mms = mms.min()

        return mms


    def max_radious(self, features, optimize=False):
        radius = max([oo.norm() for oo in features])
        if not optimize:
            radius = radius.detach()
        return radius

    def forward(self, inputs, target):
        out = self.feats(inputs)
        out = self.avgpool(out)
        features = out.view(-1, 128)
        # out = self.dropout(out)

        fm_fname = './mnist_feature_maps_baseline.npy'
        # fm_fname = './cifar10_feature_maps_pmm.npy'
        t_fname = './cifar10_targets.npy'
        if os.path.isfile(fm_fname):
            f = np.load(fm_fname)
            f = np.concatenate([f, features.cpu().numpy()])
            t = np.load(t_fname)
            t = np.concatenate([t, target.cpu().numpy()])
        else:
            f = features.cpu().numpy()
            # t = target.cpu().numpy()

        np.save('./cifar10_feature_maps_baseline', f)
        # np.save('./cifar10_feature_maps_pmm', f)
        np.save('./cifar10_targets', t)


        x = self.classifier(features)
        # out = self.avgpool(out)

        if self.optimize_mms:
            radius = self.max_radious(features, optimize=True)
            # radius = self.mean_radious(features, optimize=True)
            # mms_scaled = torch.pow(radius / self.mms_loss(x, target, optimize=True), 2)
            # mms_scaled = radius / self.mms_loss(x, target, optimize=True)
            wt2 = self.mms_loss(x, target, optimize=True)
            mms_scaled = torch.pow(radius * wt2, 2)
            # with torch.no_grad():
            #     radius = self.max_radious(features, optimize=False)
            # mms_scaled = 1. / self.mms_loss(x, target, optimize=True)
            # mms_scaled = self.mms_loss(x, optimize=True) / radius
            # mms_scaled, radius = self.logistic_mms_with_x(x, features, target, optimize=True)
        else:
            with torch.no_grad():
                radius = self.max_radious(features, optimize=False)
                wt2 = self.mms_loss(x, target, optimize=False)
                mms_scaled = torch.pow(radius * wt2, 2)
                # mms_scaled = radius / self.mms_loss(x, target, optimize=False)

        # return x, features, mms_scaled, wt2, radius
        return x, mms_scaled, radius

        # return out


def mnist(**kwargs):
    return mnist_model(**kwargs)
