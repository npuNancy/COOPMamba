import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Basic(nn.Module):
    def __init__(self, in_ch, out_ch, g=16, channel_att=False, spatial_att=False):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )

        if channel_att:
            self.att_c = nn.Sequential(
                nn.Conv2d(2*out_ch, out_ch//g, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(out_ch//g, out_ch, 1, 1, 0),
                nn.Sigmoid()
            )
        if spatial_att:
            self.att_s = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

    def forward(self, data):
        fm = self.conv1(data)
        if self.channel_att:
            fm_pool = torch.cat([F.adaptive_avg_pool2d(fm, (1, 1)), F.adaptive_max_pool2d(fm, (1, 1))], dim=1)
            att = self.att_c(fm_pool)
            fm = fm * att
        if self.spatial_att:
            fm_pool = torch.cat([torch.mean(fm, dim=1, keepdim=True), torch.max(fm, dim=1, keepdim=True)[0]], dim=1)
            att = self.att_s(fm_pool)
            fm = fm * att
        return fm


class DNM(nn.Module):
    def __init__(self, channel,kernel_size=[5], channel_att=False, spatial_att=False, upMode='bilinear'):
        super(DNM, self).__init__()
        self.upMode =upMode
        in_channel = channel
        out_channel = channel*np.sum(np.array(kernel_size)**2)
        #上采样
        self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 512, channel_att=False, spatial_att=False)
        self.conv5 = Basic(512, 512, channel_att=False, spatial_att=False)
        #下采样
        self.conv6 = Basic(512+512, 512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(256+512, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(256+128, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = nn.Conv2d(out_channel, out_channel, 1, 1, 0)
        self.kernel_pred = KernelConv(kernel_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)


    def forward(self,data):
        conv1 = self.conv1(data)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        conv6 = self.conv6(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1))
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv6, scale_factor=2, mode=self.upMode)], dim=1))
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1))
        core = self.outc(F.interpolate(conv8, scale_factor=2, mode=self.upMode))
        return self.kernel_pred(data, core)


class KernelConv(nn.Module):
    def __init__(self, kernel_size=[5]):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)

    def _convert_dict(self, core, batch_size,  channel, height, width):
        core_out = {}
        core = core.view(batch_size,  -1, channel, height, width)
        core_out[self.kernel_size[0]] = core[:,  0:self.kernel_size[0]**2, ...]
        return core_out
   

    def forward(self, data, core):
        batch_size,C,H,W = data.shape
        data =data.view(batch_size,C,H,W)
        core = self._convert_dict(core,batch_size,C,H,W)
        feature_stack = []
        pred_feature = []
        kernel = self.kernel_size[::-1]
        for index, K in enumerate(kernel):
            if not feature_stack:
                data_pad = F.pad(data, [K // 2, K // 2, K // 2, K // 2])
                for i in range(K):
                    for j in range(K):
                        feature_stack.append(data_pad[..., i:i + H, j:j + W])
                feature_stack = torch.stack(feature_stack, dim=1)
            else:
                k_diff = (kernel[index - 1] - kernel[index]) // 2
                feature_stack = feature_stack[:, :, k_diff:-k_diff, ...]
            pred_feature.append(torch.sum(
                core[K].mul(feature_stack), dim=1, keepdim=False
            ))
        pred_feature = torch.stack(pred_feature, dim=0)
        pred_feature = pred_feature.reshape(batch_size,C,H,W)
        return pred_feature
      
      
