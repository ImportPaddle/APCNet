import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from . import resnet_paddle
class ConvModule(nn.Layer):
    def __init__(self,in_channels,out_channels,kernel_size=1,padding=1,conv_cfg='conv2',norm_cfg='syncbn',act_cfg='relu'):
        super(ConvModule, self).__init__()
        self.conv=nn.Conv2D(in_channels,out_channels,kernel_size=kernel_size,padding=padding)
        self.bn=nn.BatchNorm2D(out_channels)
        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        return self.relu(x)
        
class ACM(nn.Layer):
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

        self.gla = nn.Conv2D(self.channels, self.pool_scale**2, 1, 1, 0)

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
        pooled_x = F.adaptive_avg_pool2d(x, self.pool_scale)
        # [batch_size, channels, h, w]
        x = self.input_redu_conv(x)
        # [batch_size, channels, pool_scale, pool_scale]
        pooled_x = self.pooled_redu_conv(pooled_x)
        batch_size = x.shape[0]
        # [batch_size, pool_scale * pool_scale, channels]
        
        pooled_x=paddle.reshape(pooled_x,(batch_size, self.channels,-1))
        pooled_x=pooled_x.transpose((0, 2, 1))
        # pooled_x = pooled_x.view(batch_size, self.channels,
        #                          -1).permute(0, 2, 1).contiguous()
        # [batch_size, h * w, pool_scale * pool_scale]
        # print('----')
        _=self.global_info(F.adaptive_avg_pool2d(x, 1))
        
        # print(_.shape)
        # print(x.shape)
        _=paddle.reshape(_,shape=x.shape[2:])
        print(_.shape)
        tmp=x + _
        tmp=self.gla(tmp)
    
        affinity_matrix = tmp.transpose(0, 2, 3, 1).reshape(
                                       batch_size, -1, self.pool_scale**2)
        affinity_matrix = F.sigmoid(affinity_matrix)
        # [batch_size, h * w, channels]
        z_out = paddle.matmul(affinity_matrix, pooled_x)
        # [batch_size, channels, h * w]
        z_out = z_out.permute(0, 2, 1).contiguous()
        # [batch_size, channels, h, w]
        z_out = z_out.view(batch_size, self.channels, x.size(2), x.size(3))
        z_out = self.residual_conv(z_out)
        z_out = F.relu(z_out + x)
        if self.fusion:
            z_out = self.fusion_conv(z_out)

        return z_out

class APCHead(nn.Layer):
    
    def __init__(self, pool_scales=(1, 2, 3, 6), fusion=True, **kwargs):
        super(APCHead, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))

        self.pool_scales = pool_scales
        self.fusion = fusion
        self.in_channels=2048
        self.channels=20
        self.conv_cfg=None
        self.norm_cfg='SyncBN'
        self.act_cfg='relu'
        
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
        # self.acm_modules = paddle.nn.Sequential(acm_modules)
        self.acm_modules = nn.LayerList(acm_modules)
        
        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, x):
        # x = self.backbone(inputs)
        acm_outs = [x]
        i=0
        for acm_module in self.acm_modules:
            print(i)
            acm_outs.append(acm_module(x))
        acm_outs = paddle.cat(acm_outs, dim=1)
        output = self.bottleneck(acm_outs)
        output = self.cls_seg(output)
        return output

class ResAPC(nn.Layer):
    
    def __init__(self, pool_scales=(1, 2, 3, 6), fusion=True, **kwargs):
        super(ResAPC, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.backbone=resnet_paddle.resnet101()
        self.apchead=APCHead()

    def forward(self, x):
        # x = self.backbone(inputs)
        # print(x.shape)
        x=self.backbone(x)
        print(x.shape)
        x=self.apchead(x)
        return x
    
if __name__=='__main__':
    x=paddle.normal(shape=[1,3,512,1024])
    # print(x.shape)
    # x=x.transpose(perm=[1, 0, 2,3])
    print(x.shape)
    # x=paddle.normal(shape=[1,1024])
    model=nn.Sequential()
    out=model(x)
    print(out)
    