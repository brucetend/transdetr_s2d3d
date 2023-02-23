import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from Backbone.segformer import Segformer
from Backbone.resnet_mmcv import ResNet
import random
from pathlib import Path
import os

# from decode_heads.aspp_head import ASPPHead
# from decode_heads.ham_head import LightHamHead


# from torchsummaryX import summary

from imageio import imwrite
import matplotlib.pyplot as plt
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet.models.necks import FPN
from mmcv.cnn.bricks.transformer import build_positional_encoding
from model.modules.point_sampling_panorama import get_bev_features
from mmcv.cnn.bricks import transformer



# print("**:", transformer.__file__)


normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])

# map_width = 500

class Trans4map_deformable_detr(nn.Module):
    def __init__(self, cfg, device):
        super(Trans4map_deformable_detr, self).__init__()

        # ego_feat_dim = cfg['ego_feature_dim']
        # mem_feat_dim = cfg['mem_feature_dim']
        n_obj_classes = cfg['n_obj_classes']

        #mem_update = cfg['mem_update']
        # ego_downsample = cfg['ego_downsample']

        # print('cfg__:', cfg)

        # self.mem_feat_dim = mem_feat_dim
        # self.mem_update = mem_update
        # self.ego_downsample = ego_downsample
        self.device = device
        self.device_mem = device  # cpu
        # self.device_mem = torch.device('cuda')  # cpu

        ################################################################################################################
        #### 新增 encoding 初始化！

        self.bev_h = cfg['bev_h']
        self.bev_w = cfg['bev_w']
        self.embed_dims = cfg['mem_feature_dim']
        self.bs = cfg['batch_size_every_processer']

        self.num_head = cfg["num_head"]
        self.num_point = cfg["num_point"]
        self.sampling_offsets = cfg['sampling_offsets']

        self.map_width = self.bev_w
        dtype = torch.float32


        self.bev_bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        # bev_bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

        # self.bev_queries = bev_bev_embedding.weight.to(dtype)
        self.bev_queries = torch.zeros(self.bev_h * self.bev_w, self.embed_dims)

        positional_encoding = dict(type='SinePositionalEncoding',
                                   num_feats=128,
                                   normalize=True)
        positional_encoding_bev = build_positional_encoding(positional_encoding)

        # self.bev_mask = torch.zeros((self.bs, self.bev_h, self.bev_w)).to(dtype)
        self.bev_mask = torch.zeros((self.bs, 256, 512)).to(dtype)
        ### change to pano_pos
        self.bev_pos = positional_encoding_bev(self.bev_mask).to(dtype)
        # print('self.bev_pos:', self.bev_pos.max(), self.bev_pos.min())
        

        ################################################################################################################
        ### Backbone  
        backbone_haha = 'segformer'

        if backbone_haha == 'resnet':
            self.encoder_backbone = ResNet(depth = 101)
            self.encoder_backbone.init_weights()

        elif backbone_haha == 'segformer':
            self.encoder_backbone = Segformer()
            self.pretrained_model_path = "./checkpoints/mit_b2.pth"
            # load pretrained weights
            state = torch.load(self.pretrained_model_path)
            #print('state:', state.keys())
            weights = {}
            for k, v in state.items():
                # print('key_:', k)
                weights[k] = v

            self.encoder_backbone.load_state_dict(weights, strict=False)
            
            self.linear_fuse = nn.Conv2d(64, 256, 1)  # 64


        #######################################################################################################################


        self.encoder_cfg = {'type': 'BEVFormerEncoder',
                               'num_layers': 4,
                               'pc_range': [-5, -5, -1, 5, 5, 1], # pc_range: pointcloud_range_XYZ
                               'num_points_in_pillar': 1,
                               'return_intermediate': False,
                               'transformerlayers': {'type': 'BEVFormerLayer',
                                                     'attn_cfgs': [{'type': 'SpatialCrossAttention', 'pc_range': [-5, -5, -2, 5, 5, 1],
                                                                    'deformable_attention': {'type': 'MSDeformableAttention3D', 'embed_dims': 256, 'num_points': 8, 'num_levels': 4}, 'embed_dims': 256}],
                                                     'feedforward_channels': 256,
                                                     'ffn_dropout': 0.1,
                                                     'operation_order': ('cross_attn', 'norm', 'ffn', 'norm'),
                                                     # 'operation_order': ('cross_attn', 'ffn', 'norm')
                                                     }
                            }
        self.encoder = build_transformer_layer_sequence(self.encoder_cfg )

        ####################################################### detr decoder ########################################################
        # self.decoder_cfg = {'type': 'DeformableDetrTransformerDecoder',
        #            'num_layers': 6, # 'return_intermediate': True,
        #            'return_intermediate': False,
        #            'transformerlayers': {'type': 'DetrTransformerDecoderLayer',
        #                                  'attn_cfgs': [{'type': 'MultiheadAttention', 'embed_dims': 256, 'num_heads': 8, 'dropout': 0.1},
        #                                                {'type': 'MultiScaleDeformableAttention', 'embed_dims': 256, 'num_levels': 1}],
        #                                  'feedforward_channels': 512,
        #                                  'ffn_dropout': 0.1,
        #                                  'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}

        # self.semantic_decoder = build_transformer_layer_sequence(self.decoder_cfg)
        # print('semantic_decoder:', self.semantic_decoder)

        self.reference_points = nn.Linear(self.embed_dims, 3)

        # if mem_update == 'replace':
        #     self.linlayer = nn.Linear(ego_feat_dim, mem_feat_dim)


        ########################################### segformer and decoder ##############################################
        # self.encoder = Segformer()

        # self.pretrained_model_path = "/home/zteng/Trans4Map/checkpoints/mit_b2.pth"
        # # load pretrained weights
        # state = torch.load(self.pretrained_model_path)
        # #print('state:', state.keys())
        # weights = {}
        # for k, v in state.items():
        #     # print('key_:', k)
        #     weights[k] = v

        # self.encoder.load_state_dict(weights, strict=False)

        # self.fuse = nn.Conv2d(mem_feat_dim*2, mem_feat_dim, 1, 1, 0)
        # print('self.embed_dims:', self.embed_dims)

        self.decoder = Decoder(self.embed_dims, n_obj_classes)
        
        # self.decoder = mini_Decoder_BEVSegFormer(self.embed_dims, n_obj_classes)


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)


    def mask_update(self,  # features,
                    proj_indices, masks_inliers, rgb_features):


        # state_rgb = torch.zeros((self.bs * self.bev_h * self.bev_w,  3), dtype=torch.uint8, device=self.device_mem)
        observed_masks = torch.zeros((self.bs, self.bev_h, self.bev_w), dtype=torch.bool, device=self.device)

        ################################################################################################################
        # # print('feature:', features.size())
        # feature = features[:, 0, :, :, :].to(self.device)    # torch.Size([1, 64, 128, 256])

        # # print('**mask_inliers:', masks_inliers.size())
        mask_inliers = masks_inliers[:, :, :]                  # torch.Size([1, 128, 256])

        # # print('proj_index:', proj_indices.size())
        proj_index = proj_indices                               # torch.Size([1, 250000])
        #### how to fill these TO DO!

        # m = (proj_index >= 0)  # -- (N, 500*500)
        threshold_index_m = torch.max(proj_index).item()
        m = (proj_index < threshold_index_m)


        if m.any():
            # # rgb_features = rgb_features.squeeze(0)
            # # print('size_of_rgb_features:', rgb_features.size())

            # rgb_features = rgb_features.permute(0, 2, 3, 1)

            ### 这个mask_inliers 是front_view的mask
            rgb_features = rgb_features[mask_inliers, :]
            rgb_memory = rgb_features[proj_index[m], :]
            # print('rgb_memory:', rgb_memory.size(), rgb_memory)

            # print('m_view:', m.shape)
            tmp_top_down_mask = m.view(-1)         # torch.Size([250000])
            # print('tmp_top_down_mask***:', torch.sum(tmp_top_down_mask!=0))

            #/state_rgb[tmp_top_down_mask, :] = rgb_memory.to(self.device_mem)

            ############################ rgb projection to show #############################
            #/rgb_write = torch.reshape(state_rgb,(500, 500, 3))
            # print('state_rgb:', state_rgb.size(), rgb_write.size())

            # rgb_write = rgb_write.cpu().numpy().astype(np.uint8)

                # plt.imshow(rgb_write)
                # plt.title('Topdown semantic map prediction')
                # plt.axis('off')
                # plt.show()

            ############################################################################################################
            observed_masks += m.reshape(self.bs, self.bev_w, self.bev_h)   # torch.Size([1, 500, 500])
            # print('observed_masks:', torch.sum(observed_masks==0), observed_masks.size())

        # memory = memory.view(N, map_width, map_width, self.mem_feat_dim) # torch.Size([1, 250, 250, 256])
        # memory = memory.permute(0, 3, 1, 2) # torch.Size([1, 256, 250, 250])
        # # print('memory_size:', memory.size())

        # # memory = self.fuse(memory)
        # memory = memory.to(self.device)

        # observed_masks = observed_masks.to(self.device)
        # print('observed_masks, rgb_write:', observed_masks.size(), rgb_write.shape)

        return observed_masks


    def forward(self, rgb, proj_indices, masks_inliers, rgb_no_norm, map_mask, map_heights):

        # print('rgb_rgb:', rgb.size())
        # rgb_features = rgb
        rgb_features = torch.nn.functional.interpolate(rgb, size=(512, 1024), mode = 'nearest', align_corners=None)
        
        #rgb_features = rgb_features.squeeze(0)
        rgb_features = rgb_features.unsqueeze(0)
        # print('shape_features:', rgb_features.size())


        # features = self.encoder(rgb_features)     # torch.Size([1, 1, 3, 512, 1024])
        ml_feat = self.encoder_backbone(rgb_features, is_feat=False)
        # print("ml_feat:", ml_feat.size())
        # ml_feat = self.encoder_backbone(rgb)

        # print('ml_feat_size:', ml_feat.size())
        ml_feat = self.linear_fuse(ml_feat)


        

        # ml_feat_1 = F.interpolate(ml_feat, size= (128, 256), mode='bilinear', align_corners=False)
        # ml_feat_2 = F.interpolate(ml_feat, size= (64, 128), mode='bilinear', align_corners=False)
        # ml_feat_3 = F.interpolate(ml_feat, size= (32, 64), mode='bilinear', align_corners=False)

        # ml_feat_1 = F.interpolate(ml_feat, size= (64, 128), mode='bilinear', align_corners=False)
        # ml_feat_2 = F.interpolate(ml_feat, size= (32, 64), mode='bilinear', align_corners=False)
        # ml_feat_3 = F.interpolate(ml_feat, size= (16, 32), mode='bilinear', align_corners=False)


        # print('ml_feat:', ml_feat.size(), ml_feat.dtype)  ###  ml_feat: torch.Size([1, 64, 256, 512])
        # feat_fpn = [ml_feat, ml_feat_1, ml_feat_2, ml_feat_3]
        feat_fpn = [ml_feat, ml_feat, ml_feat, ml_feat]
        # feat_fpn = ml_feat

        ##################################################################################################################################
        # in_channels = [256, 512, 1024, 2048]
        # fpn_mmdet = FPN(in_channels, 256, len(in_channels)).eval()
        # fpn_mmdet = fpn_mmdet.to(device = "cuda")
        # feat_fpn = fpn_mmdet(ml_feat)
        ##################################################################################################################################
        # print("embedding_embedding:", self.bev_queries.size())
        dtype = torch.float32
        # bev_queries = self.bev_bev_embedding.weight.to(dtype)

        bev_queries, feat_flatten, bev_h, bev_w, bev_pos, spatial_shapes, level_start_index = get_bev_features(
            feat_fpn, self.bev_queries, self.bev_h, self.bev_w, self.bev_pos)
        
        # print('bev_queries_to_show:', bev_queries)

        prev_bev = None
        shift = None
        # shift = torch.tensor([[-0.0001,  0.0416]], device='cuda:0')

        kwargs = {'img_metas': [{
            'img_shape': [(1024, 2048, 3)],
                                }],
            'num_point': self.num_point,
            'num_head': self.num_head,
            'sampling_offsets': self.sampling_offsets
                 }

        # print('feat_flatten:', feat_flatten.size())  ### torch.Size([1, 174080, 1, 256])
        observed_masks = self.mask_update(
                                            proj_indices,
                                            masks_inliers,
                                            rgb_no_norm)
        # observed_masks = map_mask
        
        ######################################Vis Masks #####################################################
        # observed_masks_write = observed_masks.squeeze(0)        
        # observed_masks_write = observed_masks_write.cpu().numpy().astype(np.uint8)
        
        # map_mask_write = map_mask.squeeze(0)        
        # map_mask_write = map_mask_write.cpu().numpy().astype(np.uint8)

        # plt.subplot(1, 2, 1)
        # plt.imshow(observed_masks_write)
        # plt.title('observed_masks')
        # plt.axis('off')

                
        # plt.subplot(1, 2, 2)
        # plt.imshow(map_mask_write)
        # plt.title('map_mask')
        # plt.axis('off')

        # plt.show()
        # run_id = random.randint(1, 100000)

        # output_dir = './test_result/detr_with_segformer/mask_to_compare/'
        # Path(output_dir).mkdir(parents=True, exist_ok=True)

        # output_name = os.path.join(output_dir, str(run_id))
            
        # plt.savefig(output_name + '.png')

        # observed_masks = map_mask
        # print('new_mask:', observed_masks.size())
        ################################################################################################################################################

        bev_embed = self.encoder(
                    bev_queries,
                    feat_flatten,                   ##### 四层feature map 拉直了来的，降采样8
                    feat_flatten,
                    bev_h=bev_h,
                    bev_w=bev_w,
                    bev_pos=bev_pos,
                    spatial_shapes=spatial_shapes,  ##### 都是feature map里来的
                    level_start_index=level_start_index,
                    prev_bev=prev_bev,
                    shift=shift,
                    map_mask = map_mask,
                    # map_mask = observed_masks, 
                    map_heights = map_heights,
                    **kwargs
                )

        # print('memory_bev_embed:', bev_embed.size())

        ############################ show feature map after encoder ###########################
        # print('bev_embed:', bev_embed.size())
        #
        # bev_embed_show = bev_embed[:,:,255]
        # bev_embed_show = bev_embed_show.cpu().detach().numpy().astype(np.uint8)
        #
        # bev_embed_show = bev_embed_show.reshape(200, 200, 1)
        # plt.imshow(bev_embed_show)
        # plt.title('Topdown semantic map prediction')
        # plt.axis('off')
        # plt.show()

        # features = features.unsqueeze(0)      # torch.Size([1, 1, 64, 128, 256])
        # predictions = F.interpolate(predictions, size=(480,640), mode="bilinear", align_corners=True)


        # ###改变位置
        # # bs = 1
        # bev_embed = bev_embed.permute(0, 2, 1)
        # bev_embed = bev_embed.view(self.bs, 256, self.bev_h, self.bev_w)
        # ### (250, 250)

        ##### 特征尺寸无法这么搞！！！
        # bev_embed = F.interpolate(bev_embed, size=(500, 500), mode="bilinear", align_corners=True)

        ################################################################################################################
        ### 新query的构建！decoder_detr
        # num_query = 2500 # 50*50, 32*32
        # dtype = torch.float
        # query_embedding = nn.Embedding(num_query, self.embed_dims * 2)
        # object_query_embeds = query_embedding.weight.to(dtype)

        # query_pos, query = torch.split(object_query_embeds, self.embed_dims, dim=1)

        # query_pos = query_pos.unsqueeze(0).expand(self.bs, -1, -1).to("cuda")
        # query = query.unsqueeze(0).expand(self.bs, -1, -1).to('cuda')


        # reference_points = self.reference_points(query_pos)
        # reference_points = reference_points.sigmoid()

        # # print('reference_points_in_decoder:', reference_points.size()) ### torch.Size([1, 900, 3])
        # # print('query_size_in_decoder:', query.size(), query_pos.size(), bev_embed.size()) ### torch.Size([1, 900, 256])
        # # print('referene_points:', reference_points.size())

        # query = query.permute(1, 0, 2)
        # query_pos = query_pos.permute(1, 0, 2)
        # bev_embed = bev_embed.permute(1, 0, 2)


        # semmap_feat, inter_references = self.semantic_decoder(query=query,
        #                                     key=None,
        #                                     value=bev_embed,
        #                                     query_pos=query_pos,
        #                                     reference_points=reference_points,
        #                                     reg_branches=None,
        #                                     cls_branches=None,
        #                                     spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
        #                                     level_start_index=torch.tensor([0], device=query.device),
        #                                     **kwargs)
        ################################################################################################################
        
        memory = bev_embed
        # print('memory:', memory.size())
        memory = memory.view(1, 500, 500, 256)
        memory = memory.permute(0, 3, 1, 2)  # torch.Size([1, 256, 500, 500])


        # semmap_feat_inter = semmap_feat_inter.reshape(1, self.embed_dims, 50, -1)
        # semmap = self.decoder(memory)
        # semmap_feat_inter = F.interpolate(semmap_feat_inter, size=(500, 500), mode="bilinear", align_corners=True)

        semmap = self.decoder(memory)
        # print('semmap:', semmap.size())

        # del memory, bev_embed, feat_fpn
        # return semmap, observed_masks, rgb_write
        return semmap, observed_masks
        ## return memory, observed_masks


class Decoder(nn.Module):
    def __init__(self, feat_dim, n_obj_classes):
        super(Decoder, self).__init__()

        self.layer = nn.Sequential(nn.Conv2d(feat_dim, 128, kernel_size=7, stride=1, padding=3, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(48),
                                    nn.ReLU(inplace=True),
                                   )

        self.obj_layer = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(48),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(48, n_obj_classes,
                                                 kernel_size=1, stride=1,
                                                 padding=0, bias=True),
                                       )

    def forward(self, memory):
        # print("memory_shape:", memory.size())
        l1 = self.layer(memory)
        out_obj = self.obj_layer(l1)
        # print("out_obj_shape:", out_obj.size())
        return out_obj


class mini_Decoder_BEVSegFormer(nn.Module):
    def __init__(self, feat_dim, n_obj_classes):
        super(mini_Decoder_BEVSegFormer, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(feat_dim, 128, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),

                                   nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=True),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),

                                   # nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                   # nn.BatchNorm2d(48),
                                   # nn.ReLU(inplace=True),
                                    )
        self.layer2 = nn.Sequential(
                                    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),

                                    nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=True),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    )

        self.obj_layer = nn.Sequential(nn.Dropout(p=0.1),
                                       nn.Conv2d(64, n_obj_classes,
                                                 kernel_size=1, stride=1,
                                                 padding=0, bias=True),
                                       )

    def forward(self, memory):
        # print("memory_shape:", memory.size())
        l1 = self.layer1(memory)
        l1_upsampling =  F.interpolate(l1, size=(200, 200), mode="bilinear", align_corners=True)

        l2 = self.layer2(l1_upsampling)
        l2_upsampling = F.interpolate(l2, size=(500,500), mode = 'bilinear', align_corners=True)


        out_obj = self.obj_layer(l2_upsampling)
        # print("out_obj_shape:", out_obj.size())
        return out_obj


