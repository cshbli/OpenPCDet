import numpy as np
import torch
import torch.nn as nn

from bstnnx_training.PyTorch.QAT import modules as bstnn

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    def __init__(self, c1, c2, k=1, s=1, p=0, bias=False, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.01)        
        self.act = nn.ReLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()        

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, quantize=True):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(input_channels, c_in_list[0], kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_in_list[0], eps=1e-3, momentum=0.01),
            nn.ReLU()
            )
        for idx in range(num_levels):
            cur_layers = [
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=1, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])

            self.upsample = nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=2, bias=True, padding=1)
            # initialize ConvTranspose2d's weight and bias, removing gradient
            nn.init.zeros_(self.upsample.bias)
            kernel = np.array([
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0]
            ])
            weight = np.zeros(self.upsample.weight.shape).astype(np.float32)
            for i in range(128):
                weight[i, i, ...] = kernel
            self.upsample.weight = nn.Parameter(torch.tensor(weight))
            self.upsample.weight.requires_grad = False
            self.upsample.bias.requires_grad = False

            self.upsample1 = nn.Conv2d(64, 64, kernel_size=1, padding=0, bias=False)
            # initialize ConvTranspose2d's weight and bias, removing gradient
            kernel = np.array([
                [1],
            ])
            weight = np.zeros(self.upsample1.weight.shape).astype(np.float32)
            for i in range(64):
                weight[i, i, ...] = kernel
            self.upsample1.weight = nn.Parameter(torch.tensor(weight))
            self.upsample1.weight.requires_grad = False

            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1:
                    self.deblocks.append(nn.Sequential(
                        # nn.Upsample(scale_factor=upsample_strides[idx], mode='nearest'),
                        # nn.Upsample(size=[248, 216], mode='nearest'),
                        self.upsample,
                        nn.Conv2d(num_filters[idx], num_upsample_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                elif stride == 1:
                    self.deblocks.append(nn.Sequential(
                        # nn.Upsample(scale_factor=upsample_strides[idx], mode='nearest'),
                        # nn.Upsample(size=[248, 216], mode='nearest'),
                        self.upsample1,
                        nn.Conv2d(num_filters[idx], num_upsample_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        #nn.Upsample(scale_factor=upsample_strides[-1], mode='nearest')
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
        self.quantize = quantize
        if quantize:
            self.cat1 = bstnn.CatChannel()
            self.quant = torch.quantization.QuantStub()

    def preprocess(self, data_dict):
        return data_dict

    def postprocess(self, data_dict):
        return data_dict

    def forward(self, spatial_features):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        # spatial_features = data_dict['spatial_features']
        if self.quantize:
            spatial_features = self.quant(spatial_features)

        ups = []
        ret_dict = {}
        x = spatial_features
        x = self.encoder(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            # x = torch.cat(ups, dim=1)
            if self.quantize:
                x = self.cat1(*ups)                
            else:
                x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        #data_dict['spatial_features_2d'] = x

        #return data_dict
        return (x,)

