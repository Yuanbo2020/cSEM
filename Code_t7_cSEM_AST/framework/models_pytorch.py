import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.pytorch_utils import do_mixup, interpolate, pad_framewise_output


def move_data_to_gpu(x, cuda):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')

        return x



#################################### mha ########################################################
import numpy as np
# transformer
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_heads = 8  # number of heads in Multi-Head Attention

class ScaledDotProductAttention_nomask(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention_nomask, self).__init__()

    def forward(self, Q, K, V, d_k=d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention_nomask(nn.Module):
    def __init__(self, d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads,
                 output_dim=d_model):
        super(MultiHeadAttention_nomask, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.layernorm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_heads * d_v, output_dim)

    def forward(self, Q, K, V, d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)

        context, attn = ScaledDotProductAttention_nomask()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        x = self.layernorm(output + residual)
        return x, attn


class EncoderLayer(nn.Module):
    def __init__(self, output_dim=d_model):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention_nomask(output_dim=output_dim)

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, input_dim, n_layers, output_dim=d_model):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(output_dim) for _ in range(n_layers)])
        self.mel_projection = nn.Linear(input_dim, d_model)

    def forward(self, enc_inputs):
        # print(enc_inputs.size())  # torch.Size([64, 54, 8, 8])
        size = enc_inputs.size()
        enc_inputs = enc_inputs.reshape(size[0], size[1], -1)
        enc_outputs = self.mel_projection(enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k=d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
#################################################################################################
class cSEM_CNNT(nn.Module):
    def __init__(self, classes_num, event_class, batchnormal=False, dropout=False):

        super(cSEM_CNNT, self).__init__()

        self.batchnormal=batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(64)

        self.dropout = dropout

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)

        self.fc_final = nn.Linear(d_model, classes_num, bias=True)
        encoder_layers = 1
        self.mha = Encoder(input_dim=256*8, n_layers=encoder_layers, output_dim=d_model)
        self.fc512_128 = nn.Linear(d_model, d_model, bias=True)

        ################# event #############################################################
        self.conv_block3_event = ConvBlock(in_channels=128, out_channels=256)

        self.fc_final_event = nn.Linear(d_model, event_class, bias=True)
        self.mha_event = Encoder(input_dim=256 * 8, n_layers=encoder_layers, output_dim=d_model)
        self.fc512_128_event = nn.Linear(d_model, d_model, bias=True)
        ################################################################################

        self.init_weight()

    def init_weight(self):
        if self.batchnormal:
            init_bn(self.bn0)
        init_layer(self.fc512_128)
        init_layer(self.fc_final)

        init_layer(self.fc512_128_event)
        init_layer(self.fc_final_event)

    def forward(self, input):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        # print(x.size())  # torch.Size([64, 1, 320, 64])
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        if self.dropout:
            x = F.dropout(x, p=0.2, training=self.training)
        # print(x.size())  torch.Size([64, 64, 160, 32])

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        if self.dropout:
            x = F.dropout(x, p=0.2, training=self.training)
        # print(x.size()) torch.Size([64, 128, 80, 16])

        x_com = x

        ############################# scene ################################################################
        x_scene = self.conv_block3(x_com, pool_size=(2, 2), pool_type='avg')
        x_scene = x_scene.transpose(1, 2)  # torch.Size([64, 40, 256, 8])
        x_scene, x_scene_self_attns = self.mha(x_scene)  # already have reshape
        # print(x_scene.size())  # torch.Size([64, 40, 512])

        x_scene = x_scene.transpose(1, 2)
        (x_scene1, _) = torch.max(x_scene, dim=2)
        x_scene2 = torch.mean(x_scene, dim=2)
        x_scene = x_scene1 + x_scene2
        # print(x_scene_scene.size()) # torch.Size([64, 512])

        x_scene_embed = F.relu_(self.fc512_128(x_scene))  # torch.Size([64, 512])
        scene = self.fc_final(x_scene_embed)
        scene_w = self.fc_final.weight

        ############################# scene ################################################################
        x_event = self.conv_block3_event(x_com, pool_size=(2, 2), pool_type='avg')
        x_event = x_event.transpose(1, 2)  # torch.Size([64, 40, 256, 8])
        x_event, x_event_self_attns = self.mha_event(x_event)  # already have reshape
        # print(x_event.size())  # torch.Size([64, 40, 512])

        x_event = x_event.transpose(1, 2)
        (x_event1, _) = torch.max(x_event, dim=2)
        x_event2 = torch.mean(x_event, dim=2)
        x_event = x_event1 + x_event2
        # print(x_event_event.size()) # torch.Size([64, 512])

        x_event_embed = F.relu_(self.fc512_128_event(x_event))  # torch.Size([64, 512])
        event_linear = self.fc_final_event(x_event_embed)
        event_w = self.fc_final_event.weight

        return scene, event_linear, scene_w, event_w, x_scene_embed, x_event_embed


class CnnT(nn.Module):
    def __init__(self, classes_num, batchnormal=False, dropout=False):

        super(CnnT, self).__init__()

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(64)

        self.dropout = dropout

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)

        self.fc_final = nn.Linear(d_model, classes_num, bias=True)
        encoder_layers = 1
        self.mha = Encoder(input_dim=256 * 8, n_layers=encoder_layers, output_dim=d_model)
        self.fc512_128 = nn.Linear(d_model, d_model, bias=True)

        ################# event #############################################################

        self.init_weight()

    def init_weight(self):
        if self.batchnormal:
            init_bn(self.bn0)
        init_layer(self.fc512_128)
        init_layer(self.fc_final)

    def forward(self, input):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        # print(x.size())  # torch.Size([64, 1, 320, 64])
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        if self.dropout:
            x = F.dropout(x, p=0.2, training=self.training)
        # print(x.size())  torch.Size([64, 64, 160, 32])

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        if self.dropout:
            x = F.dropout(x, p=0.2, training=self.training)
        # print(x.size()) torch.Size([64, 128, 80, 16])

        x_com = x

        ############################# scene ################################################################
        x_scene = self.conv_block3(x_com, pool_size=(2, 2), pool_type='avg')
        x_scene = x_scene.transpose(1, 2)  # torch.Size([64, 40, 256, 8])
        x_scene, x_scene_self_attns = self.mha(x_scene)  # already have reshape
        # print(x_scene.size())  # torch.Size([64, 40, 512])

        x_scene = x_scene.transpose(1, 2)
        (x_scene1, _) = torch.max(x_scene, dim=2)
        x_scene2 = torch.mean(x_scene, dim=2)
        x_scene = x_scene1 + x_scene2
        # print(x_scene_scene.size()) # torch.Size([64, 512])

        x_scene_embed = F.relu_(self.fc512_128(x_scene))  # torch.Size([64, 512])
        scene = self.fc_final(x_scene_embed)
        scene_w = self.fc_final.weight

        ############################# scene ################################################################

        return scene, scene_w, x_scene_embed



# AST
from functools import partial
from timm.models.layers import to_2tuple, trunc_normal_


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.proj = torch.nn.Conv2d(1, embed_dim, kernel_size=(16, 16), stride=(10, 10))

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block_AST(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x




class cSEM_AST(nn.Module):
    def __init__(self, scene_dim=10, event_dim=527, shared_layers=4, depth = 10):

        super(cSEM_AST, self).__init__()

        embed_dim = 768

        num_heads = 12
        mlp_ratio = 4.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, shared_layers)]  # stochastic depth decay rule
        self.shared_blocks = nn.ModuleList([
            Block_AST(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(shared_layers)])

        independent_layers = depth - shared_layers

        dpr_sce = [x.item() for x in torch.linspace(0, drop_path_rate, independent_layers)]
        self.scene_blocks = nn.ModuleList([
            Block_AST(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_sce[i], norm_layer=norm_layer)
            for i in range(independent_layers)])

        dpr_eve = [x.item() for x in torch.linspace(0, drop_path_rate, independent_layers)]
        self.event_blocks = nn.ModuleList([
            Block_AST(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_eve[i], norm_layer=norm_layer)
            for i in range(independent_layers)])

        self.shared_patch_embed = PatchEmbed(embed_dim=embed_dim)

        self.shared_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate, inplace=True)
        trunc_normal_(self.shared_cls_token, std=.02)

        self.apply(self._init_weights)

        self.shared_dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = 1212  # self.patch_embed.num_patches
        self.shared_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))

        trunc_normal_(self.shared_dist_token, std=.02)
        trunc_normal_(self.shared_pos_embed, std=.02)

        ########################################################################################
        my_dim = 2048

        self.norm_scene = norm_layer(embed_dim)
        self.fc_norm_scene = nn.LayerNorm(embed_dim)
        self.fc_scene_2048 = nn.Linear(embed_dim, my_dim, bias=True)

        self.fc_final_asc = nn.Linear(my_dim, scene_dim, bias=True)
        for _, module in self.fc_final_asc.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = F.normalize(module.weight, p=2, dim=1)

        self.norm_event = norm_layer(embed_dim)
        self.fc_norm_event = nn.LayerNorm(embed_dim)
        self.fc_event_2048 = nn.Linear(embed_dim, my_dim, bias=True)
        self.fc_final_event = nn.Linear(my_dim, event_dim, bias=True)

        self.my_init_weight()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def my_init_weight(self):
        init_layer(self.fc_scene_2048)
        init_layer(self.fc_event_2048)

    # @autocast()
    def forward(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        # print(x.size())
        # print(x.size())  # torch.Size([64, 1, 1024, 128])

        x = x.transpose(2, 3)
        # print(x.size())
        # print(x.size())  # torch.Size([64, 1, 128, 1024])

        B = x.shape[0]

        x = self.shared_patch_embed(x)  # torch.Size([1, 1212, 768])

        cls_tokens = self.shared_cls_token.expand(B, -1, -1)  # torch.Size([1, 1, 768])
        dist_token = self.shared_dist_token.expand(B, -1, -1)  # torch.Size([1, 1, 768])
        x = torch.cat((cls_tokens, dist_token, x), dim=1)  # torch.Size([1, 1214, 768])

        x = x + self.shared_pos_embed
        x = self.pos_drop(x)

        for shared_blk in self.shared_blocks:
            x = shared_blk(x)

        x_sce = x
        for scene_blk in self.scene_blocks:
            x_sce = scene_blk(x_sce)
        x_sce = self.norm_scene(x_sce)
        x_sce = (x_sce[:, 0] + x_sce[:, 1]) / 2

        x_sce = self.fc_norm_scene(x_sce)

        x_scene_embed = F.relu_(self.fc_scene_2048(x_sce))

        # arcsoftmax
        x_scene_embed = F.normalize(x_scene_embed, p=2, dim=1)
        x_sce = self.fc_final_asc(x_scene_embed)
        scene_w = self.fc_final_asc.weight

        x_eve = x
        for event_blk in self.event_blocks:
            x_eve = event_blk(x_eve)
        x_eve = self.norm_event(x_eve)
        x_eve = (x_eve[:, 0] + x_eve[:, 1]) / 2

        x_eve = self.fc_norm_event(x_eve)

        x_event_embed = F.relu_(self.fc_event_2048(x_eve))

        x_eve = self.fc_final_event(x_event_embed)
        event_w = self.fc_final_event.weight

        w_se = torch.matmul(scene_w, event_w.T)  # (10, 527)

        att_softmax_e = F.softmax(w_se, dim=-1)
        att_event2scene = torch.matmul(att_softmax_e, event_w)  # torch.Size([10, 2048])
        inferred_scene = torch.matmul(x_event_embed, att_event2scene.T)

        att_softmax_s = F.softmax(w_se.T, dim=-1)
        att_scene2event = torch.matmul(att_softmax_s, scene_w)  # torch.Size([527, 2048])
        inferred_event = torch.matmul(x_scene_embed, att_scene2event.T)

        return x_sce, x_eve, inferred_scene, inferred_event



