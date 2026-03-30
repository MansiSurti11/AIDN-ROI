import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# --- Components from models/common.py ---

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

# --- Components from models/arb.py ---

def grid_sample(x, offset, scale, outH=None, outW=None):
    b, _, h, w = x.size()
    if outH is None:
        grid = np.meshgrid(range(math.ceil(scale*w)), range(math.ceil(scale*h)))
    else:
        grid = np.meshgrid(range(outW), range(outH))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)

    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale - 0.5

    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) - 1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) - 1
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])

    if offset is not None:
        offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
        offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
        grid = grid + torch.cat((offset_0, offset_1), 1)
    grid = grid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, grid, padding_mode='zeros')
    return output

class SA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False, num_experts=4):
        super(SA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts
        self.bias = bias

        self.routing = nn.Sequential(
            nn.Linear(2, channels_in),
            nn.ReLU(True),
            nn.Linear(channels_in, num_experts),
            nn.Softmax(1)
        )

        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(channels_out, channels_in, kernel_size, kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

        if bias:
            self.bias_pool = nn.Parameter(torch.Tensor(num_experts, channels_out))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_pool)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)

    def forward(self, x, scale):
        scale_tensor = torch.ones(1, 1).to(x.device) / scale
        routing_weights = self.routing(torch.cat((scale_tensor, scale_tensor), 1)).view(self.num_experts, 1, 1)
        fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
        fused_weight = fused_weight.view(-1, self.channels_in, self.kernel_size, self.kernel_size)
        fused_bias = torch.mm(routing_weights.view(1, -1), self.bias_pool).view(-1) if self.bias else None
        out = F.conv2d(x, fused_weight, fused_bias, stride=self.stride, padding=self.padding)
        return out

class SA_adapt(nn.Module):
    def __init__(self, channels, num_experts=4):
        super(SA_adapt, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, 1, 1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(True),
            nn.Conv2d(channels//4, channels//4, 3, 1, 1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(True),
            nn.Conv2d(channels//4, channels//4, 3, 1, 1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(True),
            nn.Conv2d(channels//4, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.adapt = SA_conv(channels, channels, 3, 1, 1, num_experts=num_experts)

    def forward(self, x, scale):
        mask = self.mask(x)
        adapted = self.adapt(x, scale)
        return x + adapted * mask

class SCAB_downsample(nn.Module):
    def __init__(self, channels=64, num_experts=4, bias=False):
        super(SCAB_downsample, self).__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(self.channels+4, self.channels, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(self.channels, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(self.channels, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale):
        b, c, h, w = x.size()
 
        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, math.ceil(h * scale), 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, math.ceil(w * scale), 1).unsqueeze(0).float().to(x.device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale) - (torch.floor((coor_hr[1] + 0.5) / scale + 1e-3)) - 0.5

        input_coords = torch.cat((
            torch.ones_like(coor_h).expand([-1, math.ceil(scale * w)]).unsqueeze(0) / scale,
            torch.ones_like(coor_h).expand([-1, math.ceil(scale * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, math.ceil(scale * w)]).unsqueeze(0),
            coor_w.expand([math.ceil(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)
        pre_fea = grid_sample(x, None, scale) #b 64 h w
        input_cat = torch.cat([input_coords.expand([b, -1, -1, -1]), pre_fea], dim=1)

        # (2) predict filters and offsets
        embedding = self.body(input_cat) # b 64 h w
        ## offsets
        offset = self.offset(embedding)

        # (3) grid sample & spatially varying filtering
        ## grid sample
        fea0 = grid_sample(x, offset, scale)               ## b c h w
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## b c h w 1 -> b * h * w * c * 1

        ## filters: improvement content-aware
        routing_weights = self.routing(embedding) # b 4 h w
        routing_weights = routing_weights.view(b, self.num_experts, math.ceil(scale*h) * math.ceil(scale*w)).permute(0, 2, 1)      # (h*w) * n

        weight_compress = self.weight_compress.view(self.num_experts, -1) # 4 c/8 c 1 1 -> 4 c/8*c
        weight_compress = torch.matmul(routing_weights, weight_compress) # b (h*w) 4 matmul 4 c/8*c -> b h*w c/8*c
        weight_compress = weight_compress.view(b, math.ceil(scale*h), math.ceil(scale*w), self.channels//8, self.channels)# b h w c/8 c

        weight_expand = self.weight_expand.view(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(b, math.ceil(scale*h), math.ceil(scale*w), self.channels, self.channels//8)

        ## spatially varying filtering
        out = torch.matmul(weight_compress, fea) # b h w c/8 c * b h w c 1 = b h w c/8 1
        out = torch.matmul(weight_expand, out).squeeze(-1) # b h w c c/8 * b h w c/8 1 = b h w c 1

        return out.permute(0, 3, 1, 2) + fea0 # b c h w + b c h w

class SCAB_upsample(nn.Module):
    def __init__(self, channels=64, num_experts=4, bias=False):
        super(SCAB_upsample, self).__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels

        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        self.body = nn.Sequential(
            nn.Conv2d(self.channels+4, self.channels, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        self.routing = nn.Sequential(
            nn.Conv2d(self.channels, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        self.offset = nn.Conv2d(self.channels, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale, outH, outW):
        b, c, h, w = x.size()
        coor_hr = [torch.arange(0, outH, 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, outW, 1).unsqueeze(0).float().to(x.device)]
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale) - (torch.floor((coor_hr[1] + 0.5) / scale + 1e-3)) - 0.5

        input_coords = torch.cat((
            torch.ones_like(coor_h).expand([-1, outW]).unsqueeze(0) / scale,
            torch.ones_like(coor_h).expand([-1, outW]).unsqueeze(0) / scale,
            coor_h.expand([-1, outW]).unsqueeze(0),
            coor_w.expand([outH, -1]).unsqueeze(0)
        ), 0).unsqueeze(0)
        pre_fea = grid_sample(x, None, scale, outH, outW)

        input_cat = torch.cat([input_coords.expand([b, -1, -1, -1]), pre_fea], dim=1)

        embedding = self.body(input_cat)
        offset = self.offset(embedding)
        fea0 = grid_sample(x, offset, scale, outH, outW)
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)

        routing_weights = self.routing(embedding)
        routing_weights = routing_weights.view(b, self.num_experts, outH * outW).permute(0, 2, 1)

        weight_compress = self.weight_compress.view(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.view(b, outH, outW, self.channels//8, self.channels)

        weight_expand = self.weight_expand.view(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(b, outH, outW, self.channels, self.channels//8)

        out = torch.matmul(weight_compress, fea)
        out = torch.matmul(weight_expand, out).squeeze(-1)

        return out.permute(0, 3, 1, 2) + fea0

# --- Main EDRS Class ---

class EDRS(nn.Module):
    def __init__(self, mode='up', n_resblocks=16, n_feats=64, n_colors=3, rgb_range=1.0, res_scale=1, K=4, num_experts_SAconv=4, num_experts_CRM=8):
        super(EDRS, self).__init__()
        self.mode = mode
        self.n_resblocks = n_resblocks
        self.K = K
        kernel_size = 3
        act = nn.ReLU(True)
        conv = default_conv

        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        self.head = nn.Sequential(conv(n_colors, n_feats, kernel_size))
        
        self.body = nn.ModuleList([
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)
        ])
        self.body_tail = conv(n_feats, n_feats, kernel_size)

        self.sa_adapt = nn.ModuleList([
            SA_adapt(channels=n_feats, num_experts=num_experts_SAconv)
            for _ in range(n_resblocks // K)
        ])

        if mode == 'up':
            self.sa_sample = SCAB_upsample(channels=n_feats, num_experts=num_experts_CRM)
        else:
            self.sa_sample = SCAB_downsample(channels=n_feats, num_experts=num_experts_CRM)
            
        self.tail = conv(n_feats, n_colors, kernel_size)

    def forward(self, x, scale, outH=None, outW=None):
        x = self.sub_mean(x)
        x = self.head(x)

        res = x
        for i in range(self.n_resblocks):
            res = self.body[i](res)
            if (i + 1) % self.K == 0:
                res = self.sa_adapt[(i + 1) // self.K - 1](res, scale)

        res = self.body_tail(res)
        res += x

        if self.mode == 'up':
            res = self.sa_sample(res, scale, outH, outW)
        else:
            res = self.sa_sample(res, scale)
            
        x = self.tail(res)
        x = self.add_mean(x)
        return x

def load_restoration_network(weights_path, device='cpu'):
    # Restoration is 'up' mode
    model = EDRS(mode='up', n_resblocks=16, n_feats=64, n_colors=3, K=4)
    if weights_path:
        try:
            state_dict = torch.load(weights_path, map_location=device)
            # Weights might be wrapped in 'state_dict' or 'up_net'
            if 'up_net' in state_dict:
                model.load_state_dict(state_dict['up_net'])
            elif 'state_dict' in state_dict:
                 model.load_state_dict(state_dict['state_dict'])
            else:
                model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Could not load weights from {weights_path}: {e}")
    
    model.to(device)
    model.eval()
    return model

def load_encoder_network(weights_path, device='cpu'):
    # Encoder is 'down' mode
    model = EDRS(mode='down', n_resblocks=16, n_feats=64, n_colors=3, K=4)
    if weights_path:
        try:
            state_dict = torch.load(weights_path, map_location=device)
            # Weights might be wrapped in 'state_dict' or 'down_net'
            if 'down_net' in state_dict:
                model.load_state_dict(state_dict['down_net'])
            elif 'state_dict' in state_dict:
                 model.load_state_dict(state_dict['state_dict'])
            else:
                model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Could not load weights from {weights_path}: {e}")
    
    model.to(device)
    model.eval()
    return model
