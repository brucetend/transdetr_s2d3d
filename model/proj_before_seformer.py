###training_time : 2023-01-18-20-04
import torch
import torch.nn as nn
import torch.nn.functional as F
# from Backbone.segformer import Segformer
from Backbone.segformer_b0_b1 import mit_b0, mit_b1, mit_b2, mit_b4

from mmseg.models import build_backbone
from Backbone.mscan import MSCAN
from Backbone.ham_head import LightHamHead
# import Backbone.mmseg
# import Backbone.mmseg.models.backbones




class proj_before_seformer(nn.Module):
    def __init__(self, cfg, device, segformer_size = 'b2',):
        super(proj_before_seformer, self).__init__()

        self.pretrained_model_path = None

        n_obj_classes = cfg['n_obj_classes']
        self.device = device
        self.device_mem = device  # cpu

        ################################################################################################################
        #### 新增 encoding 初始化！

        # self.bev_h = cfg['bev_h']
        # self.bev_w = cfg['bev_w']
        self.embed_dims = cfg['mem_feature_dim']

        ################################################################################################################
        ### Backbone  

        if segformer_size == 'b0':
            self.encoder = mit_b0()
            self.pretrained_model_path = "./checkpoints/mit_b0.pth"
            self.decoder = Decoder(self.embed_dims, n_obj_classes)


        elif segformer_size == 'b1':
            self.encoder = mit_b1()
            # self.encoder.apply(self.weights_init)
            self.pretrained_model_path = "./checkpoints/mit_b1.pth"
            self.decoder = Decoder(self.embed_dims, n_obj_classes)


        elif segformer_size == 'b2':        
            self.encoder = mit_b2()       
            self.pretrained_model_path = "./checkpoints/mit_b2.pth"
            self.decoder = Decoder(self.embed_dims, n_obj_classes)


        elif segformer_size == 'b4':
            self.encoder = mit_b4()
            self.pretrained_model_path = "./checkpoints/mit_b4.pth"
            self.decoder = Decoder(self.embed_dims, n_obj_classes)

        else:
            # model_cfg = {'type': 'EncoderDecoder',
            #              'pretrained': None,
            #              'backbone': {'type': 'MSCAN',
            #                           'embed_dims': [32, 64, 160, 256],
            #                           'mlp_ratios': [8, 8, 4, 4],
            #                           'drop_rate': 0.0,
            #                           'drop_path_rate': 0.1,
            #                           'depths': [3, 3, 5, 2],
            #                           # 'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
            #                           'norm_cfg': {'type': 'GN', 'num_groups': 1, 'requires_grad': True},
            #
            #                           'init_cfg': {'type': 'Pretrained', 'checkpoint': 'pretrained/mscan_t.pth'}},
            #              'decode_head': {'type': 'LightHamHead',
            #                              'in_channels': [64, 160, 256],
            #                              'in_index': [1, 2, 3],
            #                              'channels': 256,
            #                              'ham_channels': 256,
            #                              'dropout_ratio': 0.1, 'num_classes': 21,
            #                              'norm_cfg': {'type': 'GN', 'num_groups': 32, 'requires_grad': True},
            #                              'align_corners': False,
            #                              'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False,
            #                                              'loss_weight': 1.0},
            #                              'ham_kwargs': {'MD_R': 16, 'SPATIAL': True, 'MD_S': 1, 'MD_D': 512,
            #                                             'TRAIN_STEPS': 6, 'EVAL_STEPS': 7, 'INV_T': 100, 'ETA': 0.9,
            #                                             'RAND_INIT': True}},
            #              # 'train_cfg': None,
            #              # 'test_cfg': {'mode': 'slide', 'crop_size': (1024, 1024), 'stride': (768, 768)}
            #              }

            model_cfg_base = {'type': 'EncoderDecoder',
                           'pretrained': None,
                           'backbone': {'type': 'MSCAN',
                                        'embed_dims': [64, 128, 320, 512],
                                        'mlp_ratios': [8, 8, 4, 4],
                                        'drop_rate': 0.0,
                                        'drop_path_rate': 0.1,
                                        'depths': [3, 3, 12, 3],
                                        'norm_cfg': {'type': 'SyncBN',
                                                     'requires_grad': True},
                                        'init_cfg': {'type': 'Pretrained',
                                                     'checkpoint': './checkpoints/mscan_b.pth'}},
                           'decode_head': {'type': 'LightHamHead',
                                           'in_channels': [128, 320, 512],
                                           'in_index': [1, 2, 3],
                                           'channels': 512,
                                           'ham_channels': 512,
                                           'dropout_ratio': 0.1,
                                           'num_classes': 21,
                                           'norm_cfg': {'type': 'GN',
                                                        'num_groups': 32,
                                                        'requires_grad': True},
                                           'align_corners': False,
                                           'loss_decode': {'type': 'CrossEntropyLoss',
                                                           'use_sigmoid': False,
                                                           'loss_weight': 1.0}},
                           # 'train_cfg': None, 'test_cfg': {'mode': 'whole'}
                           }

            model_backbone = build_backbone(model_cfg_base).backbone
            model_backbone.init_weights()
            self.encoder = model_backbone

            self.decoder = build_backbone(model_cfg_base).decode_head
            # self.decoder.init_weights()

            # self.pretrained_model_path = "./checkpoints/mscan_b.pth"


        # if self.pretrained_model_path:
        #     # load pretrained weights
        #     state = torch.load(self.pretrained_model_path)
        #     print('state:', state.keys())
        #     weights = {}
        #     for k, v in state.items():
        #         print('key_:', k)
        #         weights[k] = v
        #     self.encoder.load_state_dict(weights, strict=False)

        # self.linear_fuse = nn.Conv2d(64, 256, 1)  # 64
        self.dropout_rate = 0.1
        # self.decoder = Decoder_segformer(self.dropout_rate, n_obj_classes)
        # self.decoder = Decoder(self.embed_dims, n_obj_classes)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

            
    def memory_update(self, proj_indices, masks_inliers, rgb_features):

        
        N, C, H, W = rgb_features.shape
        # N = 1
        map_width = 500

        state = torch.zeros((N * map_width * map_width, self.embed_dims), dtype=torch.float, device=self.device_mem)
        # state_rgb = torch.zeros((N * map_width * map_width, 3), dtype=torch.uint8, device=self.device_mem)
        state_rgb = torch.zeros((N * map_width * map_width, 3), dtype=torch.float, device=self.device_mem)

        observed_masks = torch.zeros((N, map_width, map_width), dtype=torch.bool, device=self.device)

        ################################################################################################################
        # print('feature:', features.size())

        # print('**mask_inliers:', masks_inliers.size())
        mask_inliers = masks_inliers[:, :, :]                # torch.Size([1, 128, 256])

        # print('proj_index:', proj_indices.size())
        proj_index = proj_indices                            # torch.Size([1, 25000])
        #### how to fill these TO DO!

        # m = (proj_index >= 0)  # -- (N, 500*500)
        threshold_index_m = torch.max(proj_index).item()
        m = (proj_index < threshold_index_m)


        if m.any():

            rgb_features = rgb_features.permute(0, 2, 3, 1)
            rgb_features = rgb_features[mask_inliers, :]
            rgb_memory = rgb_features[proj_index[m], :]
            # print('rgb_memory:', rgb_memory.size(), rgb_memory)

            #print('m_view:', m.shape) # torch.Size([1, 250000])

            tmp_top_down_mask = m.view(-1)         # torch.Size([250000])
            # print('tmp_top_down_mask***:', torch.sum(tmp_top_down_mask!=0))

            ### state_rgb[tmp_top_down_mask, :] = (rgb_memory * 255).to(self.device_mem)
            state_rgb[tmp_top_down_mask, :] = rgb_memory.to(self.device_mem)

            ############################ rgb projection to show #############################
            rgb_write = torch.reshape(state_rgb,(500, 500, 3))
            rgb_write = rgb_write.unsqueeze(0)
            ############################################################################################################
            observed_masks += m.reshape(N, map_width, map_width)   # torch.Size([1, 500, 500])
            # print('observed_masks:', torch.sum(observed_masks==0), observed_masks.size())

        observed_masks = observed_masks.to(self.device)
        rgb_write = rgb_write.to(self.device)

        return rgb_write, observed_masks



    def forward(self, rgb, proj_indices, masks_inliers, rgb_no_norm, map_mask, map_heights):
        
        # print('rgb_rgb:', rgb.size())
        # rgb_features = torch.nn.functional.interpolate(rgb, size=(3, 512, 1024), mode = 'nearest', align_corners=None)
        rgb_features = rgb
        rgb_features = rgb_features
        # print('shape_features:', rgb_features.size())
        # rgb_features = rgb_features.permute(0, 2,3,1)

        ##############################################################################################################
        # memory = memory.permute(0, 3, 1, 2).to(rgb_features.device)

        # features = self.encoder(rgb_features)     # torch.Size([1, 1, 3, 512, 1024])
        batch_size = rgb_features.size(0)
        rgb_features = rgb_features.unsqueeze(0)
        proj_indices = proj_indices.unsqueeze(0)
        masks_inliers = masks_inliers.unsqueeze(0)

        memory = []
        observed_masks = []
        for i in range(batch_size):
            memory_a, observed_masks_a = self.memory_update(proj_indices[:, i, :],
                                                            masks_inliers[:, i,...], 
                                                            rgb_features[:, i, ...])
            memory.append(memory_a)
            observed_masks.append(observed_masks_a)

        memory = torch.cat(memory, 0)
        observed_masks = torch.cat(observed_masks, 0)




        # print("memory_size:", memory.size(), observed_masks.size())   # torch.Size([4, 500, 500, 3]) torch.Size([4, 500, 500])

        memory = memory.permute(0, 3, 1, 2)
        # memory = torch.nn.functional.interpolate(memory, size=(250, 250), mode = 'bilinear')
        memory = memory.unsqueeze(0)
        # print('memory_haha:', memory.size())

        ml_feat = self.encoder(memory)

        # ml_feat = self.encoder_backbone(rgb)
        # print("ml_feat_bevor_decoder:", len(ml_feat), ml_feat[1].size(), ml_feat[2].size())
        # ml_feat = torch.nn.functional.interpolate(ml_feat, size=(500, 500), mode = 'bilinear', align_corners=None)

        semmap = self.decoder(ml_feat)
        # print('semmap:', semmap.size())
        semmap = torch.nn.functional.interpolate(semmap, size=(500, 500), mode='nearest', align_corners=None)


        # del memory, bev_embed, feat_fpn
        # return semmap, observed_masks, rqgb_write
        # observed_masks = bev_mask

        # return semmap, observed_masks
        # print('ml_feat:', ml_feat.size())
        return semmap, observed_masks


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

class Decoder_segformer(nn.Module):
    def __init__(self, dropout_rate, n_obj_classes):
        super(Decoder_segformer, self).__init__()

        self.dropout = nn.Dropout2d(dropout_rate)
        self.linear_pred = nn.Conv2d(64, n_obj_classes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.cls = nn.Softmax(dim=1)

    def forward(self, memory):
        x = self.dropout(memory)
        x = self.linear_pred(x)
        # x = self.cls(x)
        return x