import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class Generator3(nn.Module):
    """Generator network."""

    def __init__(self, style_dim=64,num_domains=4, conv_dim=64):
        super(Generator3, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        self.bottlenneck = nn.ModuleList()
        for i in range(3):
            self.bottlenneck.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, style_dim=style_dim))

        # Up-sampling layers.
        self.up = nn.ModuleList()
        for i in range(2):
            self.up.append(ResidualBlockUP(curr_dim, style_dim))
            curr_dim = curr_dim // 2
        
        layers2 = []
        layers2.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers2.append(nn.Tanh())
        
        self.main = nn.Sequential(*layers)
        self.main2 = nn.Sequential(*layers2)

        # self.emb = nn.Sequential(nn.Embedding(num_domains,style_dim),nn.LayerNorm(normalized_shape=[style_dim],elementwise_affine=False))
        self.emb = nn.Embedding(num_domains,style_dim)

    def forward(self, x, s):
        out=self.main(x)
        for block in self.bottlenneck:
            out=block(out,s)
        for block in self.up:
            out=block(out,s)
        out=self.main2(out)

        return out

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out, style_dim):
        super(ResidualBlock, self).__init__()
        # self.main = nn.Sequential(
        #     nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
        #     AdaIN(style_dim, dim_out),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
        #     AdaIN(style_dim, dim_out))
        self.layer1=nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2=AdaIN(style_dim, dim_out)
        self.layer3=nn.ReLU(inplace=True)
        self.layer4=nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer5=AdaIN(style_dim, dim_out)

    def forward(self, x, s):
        out=self.layer1(x)
        out=self.layer2(out, s)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.layer5(out, s)
        return x + out

class ResidualBlockUP(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, curr_dim, style_dim):
        super(ResidualBlockUP, self).__init__()
        
        self.layer1=nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.layer2=AdaIN(style_dim, curr_dim // 2)
        self.layer3=nn.ReLU(inplace=True)


    def forward(self, x, s):
        out=self.layer1(x)
        out=self.layer2(out, s)
        out=self.layer3(out)
        return out

class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, repeat_num=6, style_dim=64, num_domains=4):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2


        # Bottleneck layers.
        self.bottlenneck=nn.ModuleList()
        for i in range(repeat_num):
            self.bottlenneck.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, style_dim=style_dim))
        
        
        
        # Up-sampling layers.
        self.up = nn.ModuleList()
        for i in range(2):
            self.up.append(ResidualBlockUP(curr_dim, style_dim))
            curr_dim = curr_dim // 2
        
        layers2 = []
        layers2.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers2.append(nn.Tanh())
        
        self.main = nn.Sequential(*layers)
        self.main2 = nn.Sequential(*layers2)
        self.emb = nn.Embedding(num_domains,style_dim)


    def forward(self, x, s):

        out=self.main(x)
        for block in self.bottlenneck:
            out=block(out,s)
        for block in self.up:
            out=block(out,s)
        out=self.main2(out)
        return out


class MConv(nn.Module):
    def __init__(self, num_classes=10, nc=3 ,ch=64):
        super(MConv, self).__init__()

        self.conv1 = nn.Conv2d(nc, ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv3 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv4 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.linear = nn.Linear(ch *2 * 2, num_classes)
        
        self.num=num_classes


    def forward(self, x, latent_flag = False):
        out = F.max_pool2d(F.relu(self.conv1(x)),2)
        out = F.max_pool2d(F.relu(self.conv2(out)),2)
        out = F.max_pool2d(F.relu(self.conv3(out)),2)
        out = F.max_pool2d(F.relu(self.conv4(out)),2)

        out = out.view(out.size(0), -1)
        latent_feature = out
        c = F.softmax(self.linear(out),dim=1)
        
        if latent_flag == False:
            return c
        else:
            return c, latent_feature

class pre(nn.Module):
    def __init__(self,n_classes,n_domins):
        super(pre, self).__init__()
        self.beta = nn.Parameter(data=0.33 * torch.randn(n_classes,n_domins) + 0.33, requires_grad=True)  # 10 * 4 * 1

    def forward(self, ext, label):
        # 1*4*64 b*1
        # rtx = torch.sum(torch.mul(self.beta.view(n_classes, -1, 1), ext), 1)
        rtx = torch.mm(self.beta, ext)
        ys = torch.index_select(rtx, 0, label)
        return ys

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss = 0

    if ver == 1:
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        loss = loss.abs_() / float(batch_size)
    elif ver == 2:
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
    else:
        raise ValueError('ver == 1 or 2')

    return loss

def conditional_mmd_rbf(source, target, label, num_class, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    loss = 0
    for i in range(num_class):
        source_i = source[label==i]
        target_i = target[label==i]
        loss += mmd_rbf(source_i, target_i)
    return loss / num_class

def domain_mmd_rbf(source, target, num_domain, d_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    loss = 0
    loss_overall = mmd_rbf(source, target)
    for i in range(num_domain):
        source_i = source[d_label == i]
        target_i = target[d_label == i]
        loss += mmd_rbf(source_i, target_i)
    return loss_overall - loss / num_domain

def domain_conditional_mmd_rbf(source, target, num_domain, d_label, num_class, c_label):
    loss = 0
    for i in range(num_class):
        source_i = source[c_label == i]
        target_i = target[c_label == i]
        d_label_i = d_label[c_label == i]
        loss_c = mmd_rbf(source_i, target_i)
        loss_d = 0
        for j in range(num_domain):
            source_ij = source_i[d_label_i == j]
            target_ij = target_i[d_label_i == j]
            loss_d += mmd_rbf(source_ij, target_ij)
        loss += loss_c - loss_d / num_domain

    return loss / num_class