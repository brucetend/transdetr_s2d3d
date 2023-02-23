import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from Backbone.segformer import Segformer, mit_b0_kd, mit_b1_kd


from Backbone.segformer_B4 import Segformer_B4
from torchsummary import summary
from imageio import imwrite
import matplotlib.pyplot as plt

# from mmcv.cnn import ConvModule


normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])

map_width = 500

############################################################################################################################################################
class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None):
        n,_,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            shape = x.shape[-2:]
            
            y = F.interpolate(y, shape, mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output 
        y = self.conv2(x)
        return y, x

################################################# Knowledge Review #########################################################################################
class CSF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse, len=32, reduce=16):
        super(CSF, self).__init__()
        len = max(mid_channel // reduce, len)
        self.fuse = fuse
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            #https://github.com/syt2/SKNet
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Sequential(
                nn.Conv2d(mid_channel, len, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(len),
                nn.ReLU(inplace=True)
            )
            self.fc1 = nn.Sequential(
                nn.Conv2d(mid_channel, len, kernel_size=1, stride=1, bias=False),
                nn.ReLU(inplace=True)
            )
            self.fcs = nn.ModuleList([])
            for i in range(2):
                self.fcs.append(
                    nn.Conv2d(len, mid_channel, kernel_size=1, stride=1)
                )
            self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None):
        x = self.conv1(x)
        if self.fuse:
            shape = x.shape[-2:]
            b = x.shape[0]
            y = F.interpolate(y, shape, mode="nearest")
            feas_U = [x,y]
            feas_U = torch.stack(feas_U,dim=1)
            attention = torch.sum(feas_U, dim=1)
            attention = self.gap(attention)
            if b ==1:
                attention = self.fc1(attention)
            else:
                attention = self.fc(attention)
            attention = [fc(attention) for fc in self.fcs]
            attention = torch.stack(attention, dim=1)
            attention = self.softmax(attention)
            x = torch.sum(feas_U * attention, dim=1)

        # output 
        y = self.conv2(x)
        return y, x
##########################################################################################################################################


class Trans4map_knowledge(nn.Module):
    def __init__(self, cfg, device, segformer_size = "b2", load_pretrained=True, abf_or_csf_index = "cfs"):
        super(Trans4map_knowledge, self).__init__()

        ego_feat_dim = cfg['ego_feature_dim']
        mem_feat_dim = cfg['mem_feature_dim']
        n_obj_classes = cfg['n_obj_classes']

        mem_update = cfg['mem_update']
        ego_downsample = cfg['ego_downsample']

        self.in_channels = cfg['in_channels']
        self.out_channels = cfg['out_channels']
        self.mid_channel = cfg['mid_channel']

        self.mem_feat_dim = mem_feat_dim
        self.mem_update = mem_update
        self.ego_downsample = ego_downsample
        self.device = device
        self.device_mem = device  # cpu
        # self.device_mem = torch.device('cuda')  # cpu

        ################################################################################################################
        if abf_or_csf_index == "cfs":
            csfs = nn.ModuleList()

            for idx, in_channel in enumerate(self.in_channels):
                csfs.append(CSF(in_channel, self.mid_channel, self.out_channels[idx], idx < len(self.in_channels)-1))
            self.abfs_or_csfs = csfs[::-1]
        elif abf_or_csf_index == "abf":
            abfs = nn.ModuleList()
            for idx, in_channel in enumerate(self.in_channels):
                abfs.append(ABF(in_channel, self.mid_channel, self.out_channels[idx], idx < len(self.in_channels)-1))
            self.abfs_or_csfs = abfs[::-1]

        # self.kdtype = kdtype
        # print('in_and_out_channel:', self.in_channels, self.out_channels)
        
        #### embedding linear proj #####
        # self.embed1_linearproject = nn.Linear(self.in_channels[0], self.out_channels[0])
        # self.embed2_linearproject = nn.Linear(self.in_channels[1], self.out_channels[1])
        # self.embed3_linearproject = nn.Linear(self.in_channels[2], self.out_channels[2])
        # self.embed4_linearproject = nn.Linear(self.in_channels[3], self.out_channels[3])

        ################################################################################################################

        if mem_update == 'replace':
            self.linlayer = nn.Linear(ego_feat_dim, mem_feat_dim)

        ########################################### segformer and decoder ##############################################
        
        if segformer_size == 'b0':
            self.encoder = mit_b0_kd()
            self.pretrained_model_path = "./checkpoints/mit_b0.pth"

        elif segformer_size == 'b1':
            self.encoder = mit_b1_kd()
            # self.encoder.apply(self.weights_init)
            self.pretrained_model_path = "./checkpoints/mit_b1.pth"

        elif segformer_size == 'b2':        
            self.encoder = Segformer()       
            self.pretrained_model_path = "./checkpoints/mit_b2.pth"
                
        # self.encoder = Segformer()
        # self.pretrained_model_path = "./checkpoints/mit_b2.pth"

        ################################################################################################################
        # load pretrained weights
        if load_pretrained == True:

            state = torch.load(self.pretrained_model_path)
            #print('state:', state.keys())
            weights = {}
            for k, v in state.items():
                # print('key_:', k)
                weights[k] = v

            self.encoder.load_state_dict(weights, strict=False)

        ##################################################################################################################
        # self.encoder_teacher = Segformer_B4()

        # # load pretrained weights
        # self.pretrained_model_path_b4 = "./checkpoints/model_pano_segformer/2023-01-28-18-16/ckpt_model.pkl"
        # state_teacher = torch.load(self.pretrained_model_path_b4)
        # weights_teacher = {}

        # for k, v in state_teacher.items():
        #     # print('key_:', k)
        #     weights_teacher[k] = v

        # self.encoder_teacher.load_state_dict(weights_teacher, strict=False)

        ###################################################################################################################

        # self.fuse = nn.Conv2d(mem_feat_dim*2, mem_feat_dim, 1, 1, 0)
        self.decoder = Decoder(256, n_obj_classes)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)



    def memory_update(self, features, proj_indices, masks_inliers):

        features = features.float() # torch.Size([1, 1, 64, 256, 512])

        N, C, H, W = features.shape
        # T = 1, N = 1


        if self.mem_update == 'replace':

            state = torch.zeros((N * map_width * map_width, self.mem_feat_dim), dtype=torch.float, device=self.device_mem)
            state_rgb = torch.zeros((N * map_width * map_width, 3), dtype=torch.uint8, device=self.device_mem)

        observed_masks = torch.zeros((N, map_width, map_width), dtype=torch.bool, device=self.device)

        ################################################################################################################
        # print('feature:', features.size())
        feature = features[:, :, :, :].to(self.device)    # torch.Size([1, 64, 128, 256])

        # print('**mask_inliers:', masks_inliers.size())
        mask_inliers = masks_inliers[:, :, :]                # torch.Size([1, 128, 256])

        # print('proj_index:', proj_indices.size())
        proj_index = proj_indices                            # torch.Size([1, 25000])
        #### how to fill these TO DO!

        # m = (proj_index >= 0)  # -- (N, 500*500)
        threshold_index_m = torch.max(proj_index).item()
        m = (proj_index < threshold_index_m)

        if N > 1:
            batch_offset = torch.zeros(N, device=self.device)
            batch_offset[1:] = torch.cumsum(mask_inliers.sum(dim=1).sum(dim=1), dim=0)[:-1]
            batch_offset = batch_offset.unsqueeze(1).repeat(1, map_width * map_width).long()
            proj_index += batch_offset


        if m.any():
            feature = F.interpolate(feature, size=(1024, 2048), mode="bilinear", align_corners=True)
            if self.ego_downsample:
                feature = feature[:, :, ::4, ::4]

            feature = feature.permute(0, 2, 3, 1)  # -- (N,H,W,512) # torch.Size([1, 480, 640, 64])

            feature = feature[mask_inliers, :]     # torch.Size([841877, 64])
            # print('feature_segformer:', feature.size())

            tmp_memory = feature[proj_index[m], :] # torch.Size([112116, 64])
            # print('tmp_memory:', tmp_memory.size())


            # rgb_features = rgb_features.squeeze(0)
            # print('size_of_rgb_features:', rgb_features.size())

            # rgb_features = rgb_features.permute(0, 2, 3, 1)
            # rgb_features = rgb_features[mask_inliers, :]
            # rgb_memory = rgb_features[proj_index[m], :]
            # print('rgb_memory:', rgb_memory.size(), rgb_memory)


            # print('m_view:', m.shape)
            tmp_top_down_mask = m.view(-1)         # torch.Size([250000])
            # print('tmp_top_down_mask***:', torch.sum(tmp_top_down_mask!=0))

            if self.mem_update == 'replace':
                tmp_memory = self.linlayer(tmp_memory)
                # print("tmp_memory_size:", tmp_memory.size())

                state[tmp_top_down_mask, :] = tmp_memory.to(self.device_mem)  ### torch.size([250000, 256])

                ### state_rgb[tmp_top_down_mask, :] = (rgb_memory * 255).to(self.device_mem)
                # state_rgb[tmp_top_down_mask, :] = rgb_memory.to(self.device_mem)

                ############################ rgb projection to show #############################
                # rgb_write = torch.reshape(state_rgb,(500, 500, 3))
                # print('state_rgb:', state_rgb.size(), rgb_write.size())

                # rgb_write = rgb_write.cpu().numpy().astype(np.uint8)
                #
                # plt.imshow(rgb_write)
                # plt.title('Topdown semantic map prediction')
                # plt.axis('off')
                # plt.show()

            else:
                raise NotImplementedError

            ############################################################################################################
            observed_masks += m.reshape(N, map_width, map_width)   # torch.Size([1, 500, 500])
            # print('observed_masks:', torch.sum(observed_masks==0), observed_masks.size())

            del tmp_memory
        del feature


        if self.mem_update == 'replace':
            memory = state

        memory = memory.view(N, map_width, map_width, self.mem_feat_dim) # torch.Size([1, 250, 250, 256])

        memory = memory.permute(0, 3, 1, 2) # torch.Size([1, 256, 250, 250])
        # print('memory_size:', memory.size())

        # memory = self.fuse(memory)
        memory = memory.to(self.device)
        observed_masks = observed_masks.to(self.device)

        return memory, observed_masks


    # def memory_update(self, features, proj_indices, masks_inliers, rgb_features):

    #     features = features.float() # torch.Size([1, 1, 64, 256, 512])

    #     N, T, C, H, W = features.shape
    #     # T = 1, N = 1

    #     if self.mem_update == 'replace':

    #         state = torch.zeros((N * map_width * map_width, self.mem_feat_dim), dtype=torch.float, device=self.device_mem)
    #         state_rgb = torch.zeros((N * map_width * map_width, 3), dtype=torch.uint8, device=self.device_mem)

    #     observed_masks = torch.zeros((N, map_width, map_width), dtype=torch.bool, device=self.device)

    #     ################################################################################################################
    #     # print('feature:', features.size())
    #     feature = features[:, 0, :, :, :].to(self.device)    # torch.Size([1, 64, 128, 256])

    #     # print('**mask_inliers:', masks_inliers.size())
    #     mask_inliers = masks_inliers[:, :, :]                # torch.Size([1, 128, 256])

    #     # print('proj_index:', proj_indices.size())
    #     proj_index = proj_indices                            # torch.Size([1, 25000])
    #     #### how to fill these TO DO!

    #     # m = (proj_index >= 0)  # -- (N, 500*500)
    #     threshold_index_m = torch.max(proj_index).item()
    #     m = (proj_index < threshold_index_m)

    #     if N > 1:
    #         batch_offset = torch.zeros(N, device=self.device)
    #         batch_offset[1:] = torch.cumsum(mask_inliers.sum(dim=1).sum(dim=1), dim=0)[:-1]
    #         batch_offset = batch_offset.unsqueeze(1).repeat(1, map_width * map_width).long()
    #         proj_index += batch_offset


    #     if m.any():
    #         feature = F.interpolate(feature, size=(1024, 2048), mode="bilinear", align_corners=True)
    #         if self.ego_downsample:
    #             feature = feature[:, :, ::4, ::4]

    #         feature = feature.permute(0, 2, 3, 1)  # -- (N,H,W,512) # torch.Size([1, 480, 640, 64])

    #         feature = feature[mask_inliers, :]     # torch.Size([841877, 64])
    #         # print('feature_segformer:', feature.size())

    #         tmp_memory = feature[proj_index[m], :] # torch.Size([112116, 64])
    #         # print('tmp_memory:', tmp_memory.size())


    #         # rgb_features = rgb_features.squeeze(0)
    #         # print('size_of_rgb_features:', rgb_features.size())

    #         # rgb_features = rgb_features.permute(0, 2, 3, 1)
    #         rgb_features = rgb_features[mask_inliers, :]
    #         rgb_memory = rgb_features[proj_index[m], :]
    #         # print('rgb_memory:', rgb_memory.size(), rgb_memory)


    #         # print('m_view:', m.shape)
    #         tmp_top_down_mask = m.view(-1)         # torch.Size([250000])
    #         # print('tmp_top_down_mask***:', torch.sum(tmp_top_down_mask!=0))

    #         if self.mem_update == 'replace':
    #             tmp_memory = self.linlayer(tmp_memory)
    #             # print("tmp_memory_size:", tmp_memory.size())

    #             state[tmp_top_down_mask, :] = tmp_memory.to(self.device_mem)  ### torch.size([250000, 256])

    #             ### state_rgb[tmp_top_down_mask, :] = (rgb_memory * 255).to(self.device_mem)
    #             state_rgb[tmp_top_down_mask, :] = rgb_memory.to(self.device_mem)

    #             ############################ rgb projection to show #############################
    #             rgb_write = torch.reshape(state_rgb,(500, 500, 3))
    #             # print('state_rgb:', state_rgb.size(), rgb_write.size())

    #             rgb_write = rgb_write.cpu().numpy().astype(np.uint8)
    #             #
    #             # plt.imshow(rgb_write)
    #             # plt.title('Topdown semantic map prediction')
    #             # plt.axis('off')
    #             # plt.show()

    #         else:
    #             raise NotImplementedError

    #         ############################################################################################################
    #         observed_masks += m.reshape(N, map_width, map_width)   # torch.Size([1, 500, 500])
    #         # print('observed_masks:', torch.sum(observed_masks==0), observed_masks.size())

    #         del tmp_memory
    #     del feature


    #     if self.mem_update == 'replace':
    #         memory = state

    #     memory = memory.view(N, map_width, map_width, self.mem_feat_dim) # torch.Size([1, 250, 250, 256])

    #     memory = memory.permute(0, 3, 1, 2) # torch.Size([1, 256, 250, 250])
    #     # print('memory_size:', memory.size())

    #     # memory = self.fuse(memory)
    #     memory = memory.to(self.device)
    #     observed_masks = observed_masks.to(self.device)

    #     return memory, observed_masks, rgb_write

    #################################################################################################################################
    #################################################################################################################################

    # def forward(self, features, proj_indices, masks_inliers):
    def forward(self, rgb, proj_indices, masks_inliers, rgb_no_norm):
        # print('rgb_rgb:', rgb.size())

        rgb_features = torch.nn.functional.interpolate(rgb, size=(512, 1024), mode = 'nearest', align_corners=None)
        # rgb_features = rgb
        rgb_features = rgb_features.unsqueeze(0)
        
        rgb_for_topdown = rgb
        rgb_for_topdown = rgb_for_topdown.permute(0, 2, 3, 1)
        # print('shape_features:', rgb_features.size())


        # panoramic_teacher_feature = self.encoder_teacher(rgb_features)
        # rgb_topdown_feat = torch.nn.functional.interpolate(rgb_topdown_feat, size=(500, 500), mode = 'bilinear', align_corners=None)

        student_feature, _c_b2 = self.encoder(rgb_features)     # torch.Size([1, 1, 3, 512, 1024])

        x = student_feature[::-1]  ### 翻转顺序

        results = []
        # embedproj = []

        # embed = embedding
        # embedproj = [*embedproj, self.embed1_linearproject(embed[0])]
        # embedproj = [*embedproj, self.embed2_linearproject(embed[1])]
        # embedproj = [*embedproj, self.embed3_linearproject(embed[2])]
        # embedproj = [*embedproj, self.embed4_linearproject(embed[3])]
        # # print('embedproj:', embedproj[0].size(), embedproj[1].size(), embedproj[2].size(), embedproj[3].size())  
        
        out_features, res_features = self.abfs_or_csfs[0](x[0])
        results.append(out_features)

        for features, abf_or_csf in zip(x[1:], self.abfs_or_csfs[1:]):
            out_features, res_features = abf_or_csf(features, res_features)
            results.insert(0, out_features)
            
        
        # print('results:', results[0].size(), results[1].size(),  results[2].size(),  results[3].size() )

        # embedproj = [*embedproj, self.embed1_linearproject(embed[0])]
        # embedproj = [*embedproj, self.embed2_linearproject(embed[1])]
        # embedproj = [*embedproj, self.embed3_linearproject(embed[2])]
        # embedproj = [*embedproj, self.embed4_linearproject(embed[3])]

        # print('embedproj:', embedproj[0].size())

        
        ###################################################################################################################
        features = _c_b2
        # print('features_size:', features.size())  # [1, 64, 256, 512]
        
        batch_size = features.size(0)
        features = features.unsqueeze(0)      # torch.Size([1, 1, 64, 128, 256])
        proj_indices = proj_indices.unsqueeze(0)
        masks_inliers = masks_inliers.unsqueeze(0)
        
        # predictions = F.interpolate(predictions, size=(480,640), mode="bilinear", align_corners=True)
        # memory, observed_masks_B, rgb_write = self.memory_update(features,
        #                                             proj_indices,
        #                                             masks_inliers,
        #                                             rgb_no_norm)
        memory = []
        observed_masks = []
        for i in range(batch_size):
            memory_b, observed_mask_b = self.memory_update(features[:,i,...],
                                                            proj_indices[:,i,:],
                                                            masks_inliers[:,i,...])
            memory.append(memory_b)
            observed_masks.append(observed_mask_b)
        
        memory = torch.cat(memory, 0)
        observed_masks = torch.cat(observed_masks, 0)

        semmap = self.decoder(memory)
        return semmap, observed_masks, results, _c_b2
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

