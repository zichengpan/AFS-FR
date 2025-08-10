from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor

    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # 随机drop一个完整的block，
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, image_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        """
        Map input tensor to patch.
        Args:
            image_size: input image size
            patch_size: patch size
            in_c: number of input channels
            embed_dim: embedding dimension. dimension = patch_size * patch_size * in_c
            norm_layer: The function of normalization
        """
        super().__init__()
        image_size = (image_size, image_size) # 存放图像的高和宽，后面的size[0\1]要用
        patch_size = (patch_size, patch_size) # 输入图像分割成小块（patch）时，每个小块的尺寸大小
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1]) # 计算patch的网格大小
        self.num_patches = self.grid_size[0] * self.grid_size[1] # 计算总的patch数量

        # The input tensor is divided into patches using 16x16 convolution
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size) # 使用卷积将输入图像划分成patch
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity() # 进行标准化，norm_layer可以是一个标准化函数，如果没有指定则使用恒等函数

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."  # 判断输入图像和预期图像尺寸是否相同

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2) # 先使用卷积对输入进行投影，然后对结果进行扁平化和转置
        x = self.norm(x) # 对结果进行标准化操作

        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5      # 根号d，缩放因子
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):  # act_layer=nn.GELU
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Block(nn.Module):
    def __init__(self,
                 dim,  # 模块的输入和输出特征的维度。在 Transformer 中，它通常等于每个位置的特征向量的维度。
                 num_heads, # 多头注意力机制中的头数。多头注意力机制将输入特征分为多个子空间，并在每个子空间上计算注意力权重和加权平均。
                 mlp_ratio=4., # MLP 隐藏层特征维度相对于输入特征维度的比例。MLP 是用于在每个位置上进行非线性变换的多层感知机。
                 qkv_bias=False,  # 表示是否在注意力层中使用偏置。在一些 Transformer 的变体中，注意力层的查询、键和值都会加上偏置。
                 qk_scale=None,  # 一个缩放因子，用于缩放注意力层中的查询-键（QK）向量
                 drop_ratio=0., # 用于 MLP 隐藏层和注意力层输出的 dropout 比例。dropout 是一种正则化技术，有助于防止过拟合。
                 attn_drop_ratio=0., # 用于注意力层输出的 dropout 比例。通过对注意力权重施加 dropout，可以增加模型的鲁棒性和泛化能力。
                 drop_path_ratio=0.,  # 用于随机深度（stochastic depth）的 dropout path 的比例。随机深度是一种结合残差连接和 dropout 的正则化方法，通过以一定比例丢弃模块的输入，可以减少训练中的模型过拟合。
                 act_layer=nn.GELU, # 激活函数层，默认为 GELU
                 norm_layer=nn.LayerNorm): # 归一化层
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # 随机深度（stochastic depth）的 dropout path，以下句子意味着如果 drop_path_ratio 大于 0，则使用 DropPath，否则使用 nn.Identity()
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))  # 残差连接和多头自注意力机制
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 残差连接和多层感知机（MLP：其的作用是通过多个非线性变换的层来对输入数据进行更丰富和复杂的特征提取，从而增强模型的表达能力。）

        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0.5, embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            image_size（int，tuple）：输入图像的大小。
            patch_size（int，tuple）：每个图像块（patch）的大小。
            in_c（int）：输入通道的数量。
            num_classes（int）：分类头中的类别数量。
            embed_dim（int）：嵌入维度，dim = patch_size * patch_size * in_c。
            depth（int）：Transformer 的深度。
            num_heads（int）：注意力头的数量。
            mlp_ratio（int）：MLP 隐藏维度与嵌入维度之比。
            qkv_bias（bool）：如果设置为 True，则启用 qkv 的偏置。
            qk_scale（float）：如果设置，则覆盖默认的 qk 缩放值，即 head_dim ** -0.5。
            representation_size（Optional[int]）：如果设置，将启用并设置表示层（预 Logits）的尺寸。
            distilled（bool）：模型是否包含蒸馏标记和 DeiT 模型中的头部。
            drop_ratio（float）：Dropout 的概率。
            attn_drop_ratio（float）：注意力 Dropout 的概率。
            drop_path_ratio（float）：随机深度的概率。
            embed_layer（nn.Module）：补丁嵌入层。
            norm_layer（nn.Module）：规范化层。
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes  # 分类数量
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models # 将embed_dim 的值赋值给num_features ，（可以理解为特征通道）
        self.num_tokens = 2 if distilled else 1  # 如果启用蒸馏(distilled)，则使用2个tokens，否则使用1个token
        # token 设置为2，在 VIT 模型中表示将使用两个特殊的标记来表示整个图像的全局信息，一般来说设置为1就能标记全局信息，设置为2能够引入细粒度信息
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)  # 标准化层，默认使用nn.LayerNorm
        act_layer = act_layer or nn.GELU  # 激活函数，默认使用nn.GELU

        self.patch_embed = embed_layer(image_size=image_size, patch_size=patch_size, in_c=in_c,
                                       embed_dim=embed_dim)  # 图像的嵌入层，in_c为图像的输入通道；embed_dim是将每一个分割好的块嵌入到固定长度
        num_patches = self.patch_embed.num_patches  # 图像中的图块数量

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 类别嵌入，表示整个图像的向量，可训练
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None  # 用于蒸馏的额外嵌入，如果未启用蒸馏，则为None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))  # 位置嵌入，表示每个图块的位置信息
        self.pos_drop = nn.Dropout(p=drop_ratio)  # 位置嵌入的dropout层

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # 生成随机失活率列表
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
        # dim = embed_dim：表示模块的输入和输出的维度或特征数。embed_dim 是一个变量，用来指定维度的大小。
        # num_heads = num_heads：表示自注意力机制中注意力头的数量。num_heads 是一个变量或参数，用来指定注意力头的数量。
        # mlp_ratio = mlp_ratio：表示模块中多层感知器（MLP）的隐藏层维度与输入维度之比。mlp_ratio 是一个变量或参数，用来指定隐藏层维度与输入维度的比例。
        # qkv_bias = qkv_bias：表示是否在查询、键、值的线性变换中使用偏置项。qkv_bias 是一个布尔变量或参数，用来指定是否使用偏置。
        # qk_scale = qk_scale：表示查询和键的缩放因子。qk_scale 是一个变量或参数，用来缩放查询和键。
        # drop_ratio = drop_ratio：表示模块中.dropout 操作的比例。drop_ratio 是一个变量或参数，用来控制.dropout。
        # attn_drop_ratio = attn_drop_ratio：表示自注意力机制中.dropout 操作的比例。attn_drop_ratio 是一个变量或参数，用来控制.dropout。
        # drop_path_ratio = dpr[i]：表示模块中.droppath 操作的比例。dpr[i] 是一个列表dpr中的元素，用来控制.droppath。
        # norm_layer = norm_layer：用于指定归一化层的类型或实例。norm_layer 是一个变量，用来指定归一化层的类型。
        # act_layer = act_layer：用于指定激活函数的类型或实例。act_layer 是一个变量，用来指定激活函数的类型。
            for i in range(depth)  # depth表示创建几个block

        ])  # Transformer模型的多个块组成的序列

        self.norm = norm_layer(embed_dim)  # 输出特征的标准化层

        # 如果不需要得到最后的输出结果，就不需要244-261这一部分
        if representation_size and not distilled:  # 如果指定了表示空间大小且未启用蒸馏.对应于cnn网络，即表示输出的特征图的通道数
            self.has_logits = True  # 模型具有适应表示空间的线性层
            self.num_features = representation_size  # 更新特征数量
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),  # 线性层
                ("act", nn.Tanh())  # 激活函数
            ]))  # 线性层和激活函数组成的顺序模型
        else:
            self.has_logits = False  # 模型没有适应表示空间的线性层
            self.pre_logits = nn.Identity()  # 占位符



        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()  # 分类器头部
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim,
                                       self.num_classes) if num_classes > 0 else nn.Identity()  # 用于蒸馏的分类器头部

        # 参数初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)  # 应用参数初始化函数


    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        # 将全局信息先进行抽取，然后扩展进行相加
        global_features_x = x[:, 0, :]
        x = x[:, 1:, :]
        x = self.norm(x)
        global_features_x = self.norm(global_features_x)
        # 以下为得到分类结果
        # if self.dist_token is None:
        #     return self.pre_logits(x[:, 0])
        # else:
        #     return x[:, 0], x[:, 1]
        return x, global_features_x

    def forward(self, x):
        x, cls_x = self.forward_features(x)

        return x, cls_x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(image_size=224, # 224
                              patch_size=16,
                              embed_dim=768, # 768
                              depth=12,
                              num_heads=8,
                              representation_size=768 if has_logits else None, # 768
                              num_classes=num_classes)
    # 调用VisionTransformer定义好的模型，并赋值给模型参数。
    # image_size:输入图像尺寸；patch_size：将输入图像切分成的小块。16表示切分成16×16的小块；
    # embed_dim：表示模型的嵌入维度大小，即图像块的特征向量维度
    # depth：Transformer模型的深度，表示有多少个Transformer块叠加在一起
    # num_heads：Transformer中自注意力机制的头数
    # representation_size: 输出特征向量的维度大小，如果 has_logits 为 True，则设置为 embed_dim，否则设置为 None，则表示模型不输出特征向量。
    # num_classes：分类任务的类别数量

    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(image_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)

    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(image_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)

    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(image_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)

    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(image_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)

    return model
