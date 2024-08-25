import cv2
import math

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from models.backbone import *
from libs.utils import *

from .lstn.transformer import LongShortTermTransformerBlock    
from .lstn.lstn import LSTN
from .lstn.position import PositionEmbeddingSine
from .lstn.basic import one_hot_mask, seq_to_2d

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    pe = pe.view(1, d_model * 2, height, width)
    return pe

class Deformable_Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(Deformable_Conv2d, self).__init__()
        self.deform_conv2d = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, offset, mask=None):
        out = self.deform_conv2d(x, offset, mask)
        return out

class Conv_ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(Conv_ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.conv2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out += x
        out = self.relu(out)
        return out


class conv_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class conv_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class conv1d_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(conv1d_bn_relu, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        # cfg
        self.cfg = cfg

        self.sf = self.cfg.scale_factor['img']
        self.seg_sf = self.cfg.scale_factor['seg']
        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]

        self.window_size = cfg.window_size
        self.c_feat = 64

        self.feat_embed = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
        )

        self.feat_guide = torch.nn.Sequential(
            conv_bn_relu(1, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            torch.nn.Conv2d(self.c_feat, self.c_feat, 1),
        )

        self.classifier = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=1),
            torch.nn.Conv2d(self.c_feat, 2, 1)
        )

        self.feat_embedding = torch.nn.Sequential(
            conv_bn_relu(1, self.c_feat, 3, 1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, 1, padding=2, dilation=2),
        )

        self.regressor = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, 1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, 1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, 1, padding=2, dilation=2),
        )

        kernel_size = 3
        self.offset_regression = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            torch.nn.Conv2d(self.c_feat, 2 * kernel_size * kernel_size, 1)
        )

        self.deform_conv2d = Deformable_Conv2d(in_channels=self.c_feat, out_channels=self.cfg.top_m,
                                               kernel_size=kernel_size, stride=1, padding=1)

        self.pe = positionalencoding2d(d_model=self.c_feat, height=self.cfg.height // self.seg_sf[0], width=self.cfg.width // self.seg_sf[0]).cuda()

        self.lstn = LSTN()
        self.LSAB = LongShortTermTransformerBlock()
        self.pos_emb = None
        self.pos_generator = PositionEmbeddingSine(128, normalize=True)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv_layer2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)


        self.long_term_memories = None
        self.short_term_memories_list = []
        self.short_term_memories = None
        self.pos_emb = None


    def forward_for_mask_generation(self, query_data, key_data):
        data_combined = torch.cat((query_data, key_data), dim=1)
        mask = self.mask_generator(data_combined)
        mask = torch.sigmoid(mask)
        return mask

    def forward_for_feat_aggregation(self, is_training=False):      

        key_t = f't-0'         
        query_img_feat = self.memory['img_feat'][key_t]        

        curr_emb = self.conv_layer(query_img_feat)    
        batch_size = query_img_feat.size()[0]
        enc_hw = curr_emb.size()[2]*curr_emb.size()[3]
        self.size_2d = curr_emb.size()[2:]

        lane_mask=self.memory['guide_cls']['t-1']       
        curr_one_hot_mask = one_hot_mask(lane_mask, 1) 
        if not self.memory['long']['t-1']:        
            self.pos_emb = self.lstn.get_pos_emb(curr_emb)  \
                    .expand(batch_size, -1, -1,-1).view(batch_size, -1, enc_hw).permute(2, 0, 1) 
            curr_id_emb = self.lstn.assign_identity(curr_one_hot_mask, batch_size, enc_hw)  
            self.memory['pos_emb']['t-0'] = self.pos_emb
        else:          
            curr_id_emb=None   
            self.memory['pos_emb']['t-0']=self.memory['pos_emb']['t-1']
            self.update_short_term_memory(self.lsab_curr_memorie, batch_size, enc_hw)   
        
        n, c, h, w = curr_emb.size()
        _curr_emb = curr_emb.view(n, c, h*w).permute(2, 0, 1)

        self.long_term_memories=self.memory['long']['t-1']

        lsab_embs, lsab_memories = self.LSAB(_curr_emb, self.long_term_memories, self.short_term_memories, curr_id_emb, self.memory['pos_emb']['t-0'], self.size_2d)       
        self.lsab_curr_memories, lsab_long_memories, lsab_short_memories = lsab_memories         
        self.lsab_curr_memorie = lsab_embs

        img_feat_cur = lsab_embs.permute(1, 2, 0).view(n, c, h, w)        
        self.img_feat = self.conv_layer2(img_feat_cur)

        if not self.memory['long']['t-0']:
            self.memory['long']['t-0'] = lsab_long_memories
        else:
            self.memory['long']['t-0'] = self.memory['long']['t-1']
        self.short_term_memories = self.lsab_curr_memories
        self.memory['short']['t-0'] = self.lsab_curr_memories

        return {'key_probmap': lane_mask,           
                'key_guide': lane_mask,
                'aligned_key_probmap': lane_mask,
                'aligned_key_guide': lane_mask,
                'grid': None,
                'long':self.long_term_memories,
                'pos_emb':self.pos_emb,
                'short':self.short_term_memories
                }


    def forward_for_guidance_feat(self, guide_map):
        # data = torch.cat((prob_map, guide_map), dim=1)
        feat_guide = self.feat_guide(guide_map)
        return feat_guide

    def forward_for_classification(self):
        out = self.classifier(self.img_feat)
        self.prob_map = F.softmax(out, dim=1)
        return {'seg_map_logit': out,
                'seg_map': self.prob_map[:, 1:2]}

    def forward_for_regression(self):
        b, _, _, _ = self.prob_map.shape
        feat_c = self.feat_embedding(self.prob_map[:, 1:].detach())    
        feat_c = feat_c + self.pe.expand(b, -1, -1, -1)        
        offset = self.offset_regression(feat_c)       
        x = self.regressor(feat_c)                 
        coeff_map = self.deform_conv2d(x, offset)  

        return {'coeff_map': coeff_map}

    def update_short_term_memory(self, lsab_curr_memorie, batch_size, enc_hw): 
        _lsab_curr_memorie = self.LSAB.norm2(lsab_curr_memorie) 

        curr_Q = self.LSAB.linear_Q(_lsab_curr_memorie) 
        curr_K = curr_Q  
        curr_V = _lsab_curr_memorie  

        [curr_K, curr_V] = [
            seq_to_2d(curr_K, self.size_2d),
            seq_to_2d(curr_V, self.size_2d)
        ]

        self.short_term_memories = [curr_K, curr_V]
