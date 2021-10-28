# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class ACM(nn.Module):
    """Adaptive Context Module used in APCNet.

    Args:
        pool_scale (int): Pooling scale used in Adaptive Context
            Module to extract region features.
        fusion (bool): Add one conv to fuse residual feature.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict | None): Config of conv layers.
        norm_cfg (dict | None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, pool_scale, fusion, in_channels, channels, conv_cfg,
                 norm_cfg, act_cfg):
        super(ACM, self).__init__()
        self.pool_scale = pool_scale
        self.fusion = fusion
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pooled_redu_conv = ConvModule(
            self.in_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.input_redu_conv = ConvModule(
            self.in_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.global_info = ConvModule(
            self.channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.gla = nn.Conv2d(self.channels, self.pool_scale**2, 1, 1, 0)

        self.residual_conv = ConvModule(
            self.channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        if self.fusion:
            self.fusion_conv = ConvModule(
                self.channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, x):
        """Forward function."""
        print('input acm shape ',x.shape) #[2, 2048, 64, 128]
        print('self.pool_scale',self.pool_scale) #1
        pooled_x = F.adaptive_avg_pool2d(x, self.pool_scale)
        print('pooled1_x',pooled_x.shape) #[2, 2048, 1, 1]
        # [batch_size, channels, h, w]
        x = self.input_redu_conv(x)
        print('xxxx',x.shape) #[2, 512, 64, 128]
        # [batch_size, channels, pool_scale, pool_scale]
        pooled_x = self.pooled_redu_conv(pooled_x)
        print('pooled_x pooled_redu_conv,',pooled_x.shape) #[2, 512, 1, 1]
        
        batch_size = x.size(0)
        # [batch_size, pool_scale * pool_scale, channels]
        pooled_x = pooled_x.view(batch_size, self.channels,
                                 -1).permute(0, 2, 1).contiguous()
        # [batch_size, h * w, pool_scale * pool_scale]
        print('pooled_x',pooled_x.shape) #[2, 1, 512] 1,4,9,36
        
        tmp=self.global_info(F.adaptive_avg_pool2d(x, 1))
        print('tmp.shape ',tmp.shape) #[2, 512, 1, 1]
        
        print(x.shape[2:]) #[64, 128]
        
        tmp=resize(
            tmp, size=x.shape[2:])
        
        print('reszie',tmp.shape) #[2, 512, 64, 128]
        
        tmp=x+tmp 
        print('x+tmp',tmp.shape) #[2, 512, 64, 128]
        
        tmp=self.gla(tmp)
        print('gla',tmp.shape) #[2, 1, 64, 128]
        
        affinity_matrix =tmp.permute(0, 2, 3, 1).reshape(
                                       batch_size, -1, self.pool_scale**2)
        affinity_matrix = F.sigmoid(affinity_matrix)
        # [batch_size, h * w, channels]
        z_out = torch.matmul(affinity_matrix, pooled_x)
        # [batch_size, channels, h * w]
        z_out = z_out.permute(0, 2, 1).contiguous()
        # [batch_size, channels, h, w]
        z_out = z_out.view(batch_size, self.channels, x.size(2), x.size(3))
        z_out = self.residual_conv(z_out)
        z_out = F.relu(z_out + x)
        if self.fusion:
            z_out = self.fusion_conv(z_out)

        return z_out


@HEADS.register_module()
class APCHead(BaseDecodeHead):
    """Adaptive Pyramid Context Network for Semantic Segmentation.

    This head is the implementation of
    `APCNet <https://openaccess.thecvf.com/content_CVPR_2019/papers/\
    He_Adaptive_Pyramid_Context_Network_for_Semantic_Segmentation_\
    CVPR_2019_paper.pdf>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Adaptive Context
            Module. Default: (1, 2, 3, 6).
        fusion (bool): Add one conv to fuse residual feature.
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), fusion=True, **kwargs):
        super(APCHead, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.fusion = fusion
        acm_modules = []
        for pool_scale in self.pool_scales:
            acm_modules.append(
                ACM(pool_scale,
                    self.fusion,
                    self.in_channels,
                    self.channels,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        self.acm_modules = nn.ModuleList(acm_modules)
        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        print('self.dropout_ratio apchead',self.dropout_ratio)
    def forward(self, inputs):
        """Forward function."""
        # print('self.in_channels',self.in_channels)
        # print('self.channels',self.channels)
        # print('self.conv_cfg',self.conv_cfg)
        # print('self.norm_cfg',self.norm_cfg)
        # print('self.act_cfg',self.act_cfg)
        # print(type(inputs)) #tuple
        # print(len(inputs)) #len 4
        # print(inputs[0].shape) #[2, 256, 128, 256]
        # print(inputs[1].shape) #[2, 512, 64, 128]
        # print(inputs[2].shape) #[2, 1024, 64, 128]
        # print(inputs[3].shape) #[2, 2048, 64, 128]
        x = self._transform_inputs(inputs)
        # print('x',x.shape) #index 3   [2, 2048, 64, 128]

        
        print('x.shape',x.shape) #[2, 2048, 64, 128]
        acm_outs = [x]
        
        for acm_module in self.acm_modules:
            acm_outs.append(acm_module(x))
        i=0
        for ele in acm_outs:
            print('acm module{}'.format(i), ele.shape)
            i+=1
        # x input torch.Size([2, 2048, 64, 128])
        # acm module0 torch.Size([2, 512, 64, 128])
        # acm module1 torch.Size([2, 512, 64, 128])
        # acm module2 torch.Size([2, 512, 64, 128])
        # acm module3 torch.Size([2, 512, 64, 128])
        acm_outs = torch.cat(acm_outs, dim=1)
        print('acm_outs',acm_outs.shape) #[2, 4096, 64, 128]
        output = self.bottleneck(acm_outs)
        print('bottleneck',acm_outs.shape) #[2, 4096, 64, 128]
        output = self.cls_seg(output)
        print(output.shape) #[2, 19, 64, 128]
        return output
