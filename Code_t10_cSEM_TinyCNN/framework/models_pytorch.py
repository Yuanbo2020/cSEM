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


class cSEM_TinyCNN(nn.Module):
    def __init__(self, classes_num, event_class, batchnormal=False):

        super(cSEM_TinyCNN, self).__init__()

        self.batchnormal=batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(64)

        out_channels =32
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        out_channels = 64
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.fc1 = nn.Linear(64*11, 100, bias=True)
        self.fc_final = nn.Linear(100, classes_num, bias=True)

        self.fc1_event = nn.Linear(64 * 11, 100, bias=True)
        self.fc_final_event = nn.Linear(100, event_class, bias=True)

    def init_weight(self):
        if self.batchnormal:
            init_bn(self.bn0)

        init_bn(self.bn1)
        init_bn(self.bn2)

        init_layer(self.fc1)
        init_layer(self.fc_final)

        init_layer(self.fc1_event)
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
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(5, 5))
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())  # torch.Size([64, 32, 63, 12])

        x = F.relu_(self.bn2(self.conv2(x)))
        # print(x.size())  # torch.Size([64, 64, 59, 8])
        x = F.max_pool2d(x, kernel_size=(5, 5))
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())  # torch.Size([64, 64, 11, 1])

        x = x.view(x.size()[0], -1)

        x_com = x

        ############################ scene #################################
        x_scene_embed = F.relu_(self.fc1(x_com))
        x_scene_embed = F.dropout(x_scene_embed, p=0.3, training=self.training)

        scene = self.fc_final(x_scene_embed)
        scene_w = self.fc_final.weight

        ############################ event #################################
        x_event_embed = F.relu_(self.fc1_event(x_com))
        x_event_embed = F.dropout(x_event_embed, p=0.3, training=self.training)

        event_linear = self.fc_final_event(x_event_embed)
        event_w = self.fc_final_event.weight

        w_se = torch.matmul(scene_w, event_w.T)  # (10, 527)

        att_softmax_e = F.softmax(w_se, dim=-1)
        att_event2scene = torch.matmul(att_softmax_e, event_w)  # torch.Size([10, 2048])
        inferred_scene = torch.matmul(x_event_embed, att_event2scene.T)

        att_softmax_s = F.softmax(w_se.T, dim=-1)
        att_scene2event = torch.matmul(att_softmax_s, scene_w)  # torch.Size([527, 2048])
        inferred_event = torch.matmul(x_scene_embed, att_scene2event.T)

        return scene, event_linear, inferred_scene, inferred_event


class TinyCNN(nn.Module):
    def __init__(self, classes_num, batchnormal=False):

        super(TinyCNN, self).__init__()

        self.batchnormal=batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(64)

        out_channels =32
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        out_channels = 64
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.fc1 = nn.Linear(64*11, 100, bias=True)
        self.fc_final = nn.Linear(100, classes_num, bias=True)

    def init_weight(self):
        if self.batchnormal:
            init_bn(self.bn0)

        init_bn(self.bn1)
        init_bn(self.bn2)

        init_layer(self.fc1)
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
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(5, 5))
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())  # torch.Size([64, 32, 63, 12])

        x = F.relu_(self.bn2(self.conv2(x)))
        # print(x.size())  # torch.Size([64, 64, 59, 8])
        x = F.max_pool2d(x, kernel_size=(5, 5))
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())  # torch.Size([64, 64, 11, 1])

        x = x.view(x.size()[0], -1)

        x_com = x

        ############################ scene #################################
        x_scene_embed = F.relu_(self.fc1(x_com))
        x_scene_embed = F.dropout(x_scene_embed, p=0.3, training=self.training)

        scene = self.fc_final(x_scene_embed)
        scene_w = self.fc_final.weight


        return scene, scene_w, x_scene_embed






