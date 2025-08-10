import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torchvision.transforms as transforms
from .backbones import ResNet,tran_encoder


model_vit_path = r'E:\work\AFS-FR\vit_base_patch16_224_in21k.pth'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class FRN(nn.Module):
    
    def __init__(self,way=None,shots=None,resnet=False,is_pretraining=False,num_cat=None):
        
        super().__init__()

        if resnet:
            num_channel = 640
            num_channel_tran = 768
            self.feature_extractor = ResNet.resnet12()
            self.feature_extractor = nn.DataParallel(self.feature_extractor, device_ids=[0])

            self.feature_extractor_tran = tran_encoder.vit_base_patch16_224_in21k()
            self.feature_extractor_tran.load_state_dict(torch.load(model_vit_path), strict=True)
            self.feature_extractor_tran = nn.DataParallel(self.feature_extractor_tran, device_ids=[0])


        self.shots = shots
        self.way = way
        self.resnet = resnet
        self.transform_84 = transforms.Resize((84,84))
        self.cov1 = nn.Conv2d(num_channel_tran, num_channel, kernel_size=1, stride=1)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.maxpool2 = nn.MaxPool2d(3, stride=1, padding=0)
        # self.ECA = ECANet(in_channels=num_channel)
        self.g = np.random.random() * (1/20)


        # number of channels for the feature map, correspond to d in the paper
        self.d = num_channel
        
        # temperature scaling, correspond to gamma in the paper
        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)
        
        # H*W=5*5=25, resolution of feature map, correspond to r in the paper
        self.resolution = 25

        # correpond to [alpha, beta] in the paper
        # if is during pre-training, we fix them to 0
        self.r = nn.Parameter(torch.zeros(2),requires_grad=not is_pretraining)
        self.w1 = nn.Parameter(torch.FloatTensor([1]),requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor([1]),requires_grad=True)
        self.w3 = nn.Parameter(torch.FloatTensor([1]),requires_grad=True)
        self.w4 = nn.Parameter(torch.FloatTensor([1]),requires_grad=True)
        self.w5 = nn.Parameter(torch.FloatTensor([0.1]), requires_grad=True)
        if is_pretraining:
            # number of categories during pre-training
            self.num_cat = num_cat
            # category matrix, correspond to matrix M of section 3.6 in the paper
            self.cat_mat = nn.Parameter(torch.randn(self.num_cat,self.resolution,self.d),requires_grad=True)

        ngpu = 3
        if ngpu > 1:
            self.feature_extractor = nn.DataParallel(self.feature_extractor, device_ids=[0])
            self.feature_extractor_tran = nn.DataParallel(self.feature_extractor_tran, device_ids=[0])
    

    def get_feature_map(self,inp):
        inp_tran = self.transform_84(inp)

        batch_size = inp.size(0)
        feature_map = self.feature_extractor.module(inp_tran)
        feature_map_tran, cls_feature = self.feature_extractor_tran.module(inp)
        b, c_t, hw = feature_map_tran.size()
        feature_map_tran = feature_map_tran.permute(0,2,1).view(batch_size, hw, 14, 14)

        ## AFSM
        feature_map_tran = self.maxpool1(feature_map_tran)
        feature_map_tran = self.maxpool2(feature_map_tran)
        cls_feature = cls_feature.unsqueeze(-1).unsqueeze(-1)
        feature_map_tran = self.cov1(feature_map_tran)
        cls_feature = self.cov1(cls_feature)
        cls_feature = self.w5 * cls_feature


        feature_map_avg = self.maxpool(feature_map)
        # #
        z_cls_feature = torch.cat((feature_map_avg, cls_feature), dim=1)
        z_cls_feature = z_cls_feature.view(z_cls_feature.size(0), z_cls_feature.size(1))
        w_count = F.softmax(z_cls_feature, dim=1)
        sum1 = w_count[:, :640].sum(dim=1)
        sum2 = w_count[:, 640:].sum(dim=1)
        feature_map = feature_map + self.g * feature_map
        feature_map_tran = feature_map_tran + self.g * feature_map_tran
        feature_map = self.w1 * feature_map * sum1.view(w_count.size(0), 1, 1, 1)
        feature_map_tran = self.w2 * feature_map_tran * sum2.view(w_count.size(0), 1, 1, 1)
        feature_map = feature_map.view(batch_size,self.d,-1).permute(0,2,1).contiguous()
        feature_map_tran = feature_map_tran.view(batch_size, self.d, -1).permute(0, 2, 1).contiguous()
        
        return feature_map, feature_map_tran
    

    def get_recon_dist(self,query,support,alpha,beta,Woodbury=False):
    # query: way*query_shot*resolution, d
    # support: way, shot*resolution , d
    # Woodbury: whether to use the Woodbury Identity as the implementation or not

        # correspond to kr/d in the paper
        reg = support.size(1)/support.size(2)
        
        # correspond to lambda in the paper
        lam = reg*alpha.exp()+1e-6

        # correspond to gamma in the paper
        rho = beta.exp()

        st = support.permute(0,2,1) # way, d, shot*resolution

        if Woodbury:
            # correspond to Equation 10 in the paper
            
            sts = st.matmul(support) # way, d, d
            m_inv = (sts+torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse() # way, d, d
            hat = m_inv.matmul(sts) # way, d, d
        
        else:
            # correspond to Equation 8 in the paper
            
            sst = support.matmul(st) # way, shot*resolution, shot*resolution
            m_inv = (sst+torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)).inverse() # way, shot*resolution, shot*resolutionsf 
            hat = st.matmul(m_inv).matmul(support) # way, d, d

        Q_bar = query.matmul(hat).mul(rho) # way, way*query_shot*resolution, d

        dist = (Q_bar-query.unsqueeze(0)).pow(2).sum(2).permute(1,0) # way*query_shot*resolution, way
        
        return dist

    
    def get_neg_l2_dist(self,inp,way,shot,query_shot,return_support=True,val=False):
        
        resolution = self.resolution
        d = self.d
        alpha = self.r[0]
        beta = self.r[1]

        feature_map, feature_map_tran = self.get_feature_map(inp)
        #
        support = feature_map[:way*shot].view(way, shot*resolution , d)
        query = feature_map[way*shot:].view(way*query_shot*resolution, d)

        support_tran = feature_map_tran[:way*shot].view(way, shot*resolution , d)
        query_tran = feature_map_tran[way*shot:].view(way*query_shot*resolution, d)

        recon_dist = self.get_recon_dist(query=query,support=support,alpha=alpha,beta=beta) # way*query_shot*resolution, way
        neg_l2_dist = recon_dist.neg().view(way*query_shot,resolution,way).mean(1) # way*query_shot, way
        recon_dist_tran = self.get_recon_dist(query=query_tran,support=support_tran,alpha=alpha,beta=beta) # way*query_shot*resolution, way
        neg_l2_dist_tran = recon_dist_tran.neg().view(way*query_shot,resolution,way).mean(1) # way*query_shot, way
        neg_l_dist = (self.w3 * neg_l2_dist) * (self.w4 * neg_l2_dist_tran)
        neg_l_dist = torch.sqrt(neg_l_dist)

        neg_l_dist = torch.neg(neg_l_dist)
        if return_support:
            return neg_l_dist
        else:
            return neg_l_dist


    def meta_test(self,inp,way,shot,query_shot):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                        way=way,
                                        shot=shot,
                                        query_shot=query_shot,
                                        return_support=False,
                                        val=True)

        _,max_index = torch.max(neg_l2_dist,1)

        return max_index
    

    def forward_pretrain(self,inp):

        feature_map, feature_map_tran = self.get_feature_map(inp)
        batch_size = feature_map.size(0)

        feature_map = feature_map.view(batch_size*self.resolution,self.d)
        feature_map_tran = feature_map_tran.view(batch_size * self.resolution, self.d)
        
        alpha = self.r[0]
        beta = self.r[1]
        
        recon_dist = self.get_recon_dist(query=feature_map,support=self.cat_mat,alpha=alpha,beta=beta) # way*query_shot*resolution, way

        neg_l2_dist = recon_dist.neg().view(batch_size,self.resolution,self.num_cat).mean(1) # batch_size,num_cat
        recon_dist_tran = self.get_recon_dist(query=feature_map_tran, support=self.cat_mat, alpha=alpha, beta=beta)  # way*query_shot*resolution, way

        neg_l2_dist_tran = recon_dist_tran.neg().view(batch_size, self.resolution, self.num_cat).mean(1)  # batch_size,num_cat
        neg_l_dist = (self.w3 * neg_l2_dist) * (self.w4 * neg_l2_dist_tran)
        neg_l_dist = torch.sqrt(neg_l_dist)

        neg_l_dist = torch.neg(neg_l_dist)
        logits = neg_l_dist*self.scale
        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction

    def forward(self, inp):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                                    way=self.way,
                                                    shot=self.shots[0],
                                                    query_shot=self.shots[1],
                                                    return_support=True)

        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction