import configparser
import sys, os, argparse

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

from framework.models_pytorch import *
from framework.pytorch_utils import count_parameters, count_flops

import warnings

warnings.filterwarnings('ignore')
import torch
from thop import profile


def main(argv):
    sample_rate = 32000
    clip_samples = sample_rate * 10
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000
    batch_size = 32
    early_stop = 1000000
    classes_num = 10
    event_class = 527

    model = MlhE_AST(classes_num, event_class)
    # print(model)

    params_num = count_parameters(model)
    print('Parameters num: {}'.format(params_num))

    from torchprofile import profile_macs
    input1 = torch.randn(1, 1024, 128)
    mac = profile_macs(model, input1)
    print('MAC = ' + str(mac / 1000 ** 3) + ' G')


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















