import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


        return scene


