import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
# from Backbone.segformer import Segformer

from Backbone.segformer_b0_b1 import mit_b0, mit_b1, mit_b2, mit_b4


from torchsummary import summary
from imageio import imwrite
import matplotlib.pyplot as plt

# from mmcv.cnn import ConvModule


normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

# depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])

map_width = 500
############################################################################################################################################################
class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse = True):
        super(ABF, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3,  stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True)
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
        # nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None):
        n,_,h,w = x.shape
        # transform student features
        # x = self.conv1(x)
        # y = self.conv1(y)

        if self.att_conv is not None:
            # upsample residual features
            # y = F.interpolate(y, (shape,shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output 
        y = self.conv2(x)
        return y


################################################# Knowledge Review #########################################################################################
class Sknet(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse=True, len=32, reduce=16):
        super(Sknet, self).__init__()
        len = max(mid_channel // reduce, len)
        self.fuse = fuse
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
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
        y = self.conv1(y)

        if self.fuse:
            shape = x.shape[-2:]
            b = x.shape[0]
            # y = F.interpolate(y, shape, mode="nearest")
            feas_U = [x,y]
            
            feas_U = torch.stack(feas_U,dim=1)
            attention = torch.sum(feas_U, dim=1)  #  torch.Size([1, 2, 128, 500, 500])
            attention = self.gap(attention)

            if b ==1:
                attention = self.fc1(attention)
            else:
                attention = self.fc(attention)

            attention = [fc(attention) for fc in self.fcs]
            attention = torch.stack(attention, dim=1)
            attention = self.softmax(attention)
            # print('attention1:', attention.size())


            x = torch.sum(feas_U * attention, dim=1)

        # output 
        y = self.conv2(x)
        return y
##########################################################################################################################################


class Trans4map_segformer_2branch(nn.Module):
    def __init__(self, cfg, device, segformer_size = "b2", sknet_index = True):
        super(Trans4map_segformer_2branch, self).__init__()

        ego_feat_dim = cfg['ego_feature_dim']
        mem_feat_dim = cfg['mem_feature_dim']
        n_obj_classes = cfg['n_obj_classes']

        mem_update = cfg['mem_update']
        ego_downsample = cfg['ego_downsample']

        self.ego_feature_dim = ego_feat_dim
        self.mem_feat_dim = mem_feat_dim
        self.mem_update = mem_update
        self.ego_downsample = ego_downsample
        self.device = device
        self.device_mem = device  # cpu
        # self.device_mem = torch.device('cuda')  # cpu

        self.in_channel = cfg['in_channels']
        self.out_channel = cfg['out_channels']
        self.mid_channel = cfg['mid_channel']
        self.sknet_index = sknet_index


        if mem_update == 'replace':
            self.linlayer = nn.Linear(ego_feat_dim, mem_feat_dim)

        ########################################### segformer and decoder ##############################################
        # self.encoder = mit_b2()
        # self.pretrained_model_path = "./checkpoints/mit_b2.pth"

        if segformer_size == 'b0':
            self.encoder = mit_b0
            self.pretrained_model_path = "./checkpoints/mit_b0.pth"

        elif segformer_size == 'b1':
            self.encoder = mit_b1()
            # self.encoder.apply(self.weights_init)
            self.pretrained_model_path = "./checkpoints/mit_b1.pth"

        elif segformer_size == 'b2':        
            self.encoder = mit_b2()      
            self.pretrained_model_path = "./checkpoints/mit_b2.pth"

        elif segformer_size == 'b4':
            self.encoder = mit_b4()
            self.pretrained_model_path = "./checkpoints/mit_b4.pth"

        # load pretrained weights
        state = torch.load(self.pretrained_model_path)
        #print('state:', state.keys())
        weights = {}
        for k, v in state.items():
            # print('key_:', k)
            weights[k] = v

        self.encoder.load_state_dict(weights, strict=False)

        ##################################################################################################################
        self.encoder_topdown = mit_b0()
        self.pretrained_model_path_topdown = "./checkpoints/mit_b0.pth"

        # load pretrained weights
        state_topdown = torch.load(self.pretrained_model_path_topdown)
        #print('state:', state.keys())
        weights_top_down = {}
        for k, v in state_topdown.items():
            # print('key_:', k)
            weights_top_down[k] = v

        self.encoder_topdown.load_state_dict(weights_top_down, strict=False)
        ###################################################################################################################
        if self.sknet_index == True:
            # self.sknet = Sknet(self.in_channel, self.mid_channel, self.out_channel)
            self.abf = ABF(self.in_channel, self.mid_channel, self.out_channel)
        ###################################################################################################################

        # self.fuse = nn.Conv2d(mem_feat_dim*2, mem_feat_dim, 1, 1, 0)
        # self.decoder = Decoder(320, n_obj_classes)
        # self.decoder = Decoder(128, n_obj_classes)
        self.decoder = Decoder(64, n_obj_classes)


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

            # state = torch.zeros((N * map_width * map_width, self.mem_feat_dim), dtype=torch.float, device=self.device_mem)
            state = torch.zeros((N * map_width * map_width, self.ego_feature_dim), dtype=torch.float, device=self.device_mem)

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
                
                # tmp_memory = self.linlayer(tmp_memory)
                ### linear proj from 64 to 256
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

        #memory = memory.view(N, map_width, map_width, self.mem_feat_dim) # torch.Size([1, 250, 250, 256])
        memory = memory.view(N, map_width, map_width, self.ego_feature_dim)
        # ego_feature_dim
        memory = memory.permute(0, 3, 1, 2) # torch.Size([1, 256, 250, 250])
        # print('memory_size:', memory.size())

        # memory = self.fuse(memory)
        memory = memory.to(self.device)
        observed_masks = observed_masks.to(self.device)

        return memory, observed_masks


    #############################################################################################################################
    def topdown_mapping_rgb(self, proj_indices, masks_inliers, rgb_features):

        
        N, C, H, W = rgb_features.shape
        # N = 1
        map_width = 500

        state = torch.zeros((N * map_width * map_width, self.mem_feat_dim), dtype=torch.float, device=self.device_mem)
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

            # rgb_features = rgb_features.permute(0, 2, 3, 1)
            rgb_features = rgb_features[mask_inliers, :]
            rgb_memory = rgb_features[proj_index[m], :]
            # print('rgb_memory:', rgb_memory.size(), rgb_memory)

            #print('m_view:', m.shape) # torch.Size([1, 250000])

            tmp_top_down_mask = m.view(-1)         # torch.Size([250000])
            # print('tmp_top_down_mask***:', torch.sum(tmp_top_down_mask!=0))

            ### state_rgb[tmp_top_down_mask, :] = (rgb_memory * 255).to(self.device_mem)
            state_rgb[tmp_top_down_mask, :] = rgb_memory.to(self.device_mem)

            ############################ rgb projection to show #############################
            rgb_write = torch.reshape(state_rgb,(1, 500, 500, 3))

            ############################################################################################################
            observed_masks += m.reshape(N, map_width, map_width)   # torch.Size([1, 500, 500])
            # print('observed_masks:', torch.sum(observed_masks==0), observed_masks.size())

        observed_masks = observed_masks.to(self.device)
        rgb_write = rgb_write.to(self.device)

        return rgb_write, observed_masks
    
    #################################################################################################################################


    # def forward(self, features, proj_indices, masks_inliers):
    def forward(self, rgb, proj_indices, masks_inliers, rgb_no_norm):
        # print('rgb_rgb:', rgb.size())

        rgb_features = torch.nn.functional.interpolate(rgb, size=(512, 1024), mode = 'bilinear', align_corners=None)
        rgb_features = rgb_features.unsqueeze(0)
        
        rgb_for_topdown = rgb
        rgb_for_topdown = rgb_for_topdown.permute(0, 2, 3, 1)  # torch.Size([4, 1024, 2048, 3])
        
        
        batch_size = rgb_for_topdown.size(0)
        rgb_for_topdown = rgb_for_topdown.unsqueeze(0)      # torch.Size([1, 2, 64, 128, 256])
        proj_indices = proj_indices.unsqueeze(0)
        masks_inliers = masks_inliers.unsqueeze(0)




        rgb_topdown = []
        observed_masks_A = []
        for i in range(batch_size):
            rgb_topdown_a, observed_masks_A_a = self.topdown_mapping_rgb(
                                        proj_indices[:,i,:],
                                        masks_inliers[:,i,...],
                                        rgb_for_topdown[:, i, ...])
            rgb_topdown.append(rgb_topdown_a)
            observed_masks_A.append(observed_masks_A_a)
        
        rgb_topdown = torch.cat(rgb_topdown, 0)
        observed_masks_A = torch.cat(observed_masks_A, 0)
        

        # print('shape_features:', rgb_features.size(), rgb_topdown.size()) #  torch.Size([1, 4, 3, 512, 1024]) torch.Size([4, 500, 500, 3])
        rgb_topdown = rgb_topdown.unsqueeze(0)
        rgb_topdown = rgb_topdown.permute(0, 1, 4, 2, 3)
        # features = self.encoder(rgb_features)     # torch.Size([1, 1, 3, 512, 1024])

        rgb_topdown_feat = self.encoder_topdown(rgb_topdown)
        # print('rgb_topdown_feat0:', rgb_topdown_feat.size())
        rgb_topdown_feat = torch.nn.functional.interpolate(rgb_topdown_feat, size=(500, 500), mode = 'bilinear', align_corners=None)
        # print('rgb_topdown_feat1:', rgb_topdown_feat.size())

        # print("rgb_topdown:", rgb_topdown.size(), rgb_features.size())

        features = self.encoder(rgb_features)     # torch.Size([1, 1, 3, 512, 1024])
        # print('features_size:', features.size())  # torch.Size([4, 64, 128, 256])
        
        features = features.unsqueeze(0)      # torch.Size([1, 1, 64, 128, 256])
        
        memory = []
        observed_masks = []
        for i in range(batch_size):
            memory_B, observed_masks_B = self.memory_update(features[:,i,...],
                                                        proj_indices[:,i,:],
                                                        masks_inliers[:,i,...]
                                                        )
            memory.append(memory_B)
            observed_masks.append(observed_masks_B)

        memory = torch.cat(memory, 0)
        observed_masks = torch.cat(observed_masks, 0)

            
        if self.sknet_index == True:
            # memory_sknet = self.sknet(memory, rgb_topdown_feat)
            # memory = torch.cat((memory, memory_sknet), 1)
            memory = self.abf(memory, rgb_topdown_feat)
        else:
            # memory = torch.cat((memory, rgb_topdown_feat), 1)
            memory = memory +  rgb_topdown_feat
            # print("memory_memory:", memory.size())

        semmap = self.decoder(memory)
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

