import numpy as np
import time
import torch
import torch.nn as nn


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).

    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)

    Returns:
      out: (batch_size, ...)
    """
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out
    

def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]


def forward(model, generator, return_input=False, 
    return_target=False, using_mel_in_models=False, using_stft_in_models=False):

    output_dict = {}
    device = next(model.parameters()).device
    time1 = time.time()

    # Forward data to a model in mini-batches
    for n, batch_data_dict in enumerate(generator):
        print(n)
        if using_mel_in_models:
            batch_waveform = move_data_to_device(batch_data_dict['logmel'], device)
        elif using_stft_in_models:
            batch_waveform = move_data_to_device(batch_data_dict['stft'], device)
        else:
            batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)
        
        with torch.no_grad():
            model.eval()
            batch_output = model(batch_waveform)

        append_to_dict(output_dict, 'audio_name', batch_data_dict['audio_name'])

        append_to_dict(output_dict, 'clipwise_output', 
            batch_output['clipwise_output'].data.cpu().numpy())

        if 'segmentwise_output' in batch_output.keys():
            append_to_dict(output_dict, 'segmentwise_output', 
                batch_output['segmentwise_output'].data.cpu().numpy())

        if 'framewise_output' in batch_output.keys():
            append_to_dict(output_dict, 'framewise_output', 
                batch_output['framewise_output'].data.cpu().numpy())
            
        if return_input:
            append_to_dict(output_dict, 'waveform', batch_data_dict['waveform'])
            
        if return_target:
            if 'target' in batch_data_dict.keys():
                append_to_dict(output_dict, 'target', batch_data_dict['target'])

        if n % 10 == 0:
            print(' --- Inference time: {:.3f} s / 10 iterations ---'.format(
                time.time() - time1))
            time1 = time.time()

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict


import h5py
def forward_hyb_output_stft_logmel(model, generator, audios_num, sample_rate, classes_num,
                                    waveforms_hdf5_path,
                                   return_input=False, return_target=False, output_int16=False):
    output_dict = {}
    device = next(model.parameters()).device
    time1 = time.time()

    frame_num = 1001
    mel_bins = 64

    if sample_rate==16000:
        stft_bins = 257
    elif sample_rate==32000:
        stft_bins = 513

    with h5py.File(waveforms_hdf5_path, 'w') as hf:
        hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S20')
        if output_int16:
            hf.create_dataset('logmel', shape=((audios_num, frame_num, mel_bins)), dtype=np.int16)
            hf.create_dataset('stft', shape=((audios_num, frame_num, stft_bins)), dtype=np.int16)
        else:
            hf.create_dataset('logmel', shape=((audios_num, frame_num, mel_bins)), dtype=np.float32)
            hf.create_dataset('stft', shape=((audios_num, frame_num, stft_bins)), dtype=np.float32)
        hf.create_dataset('target', shape=((audios_num, classes_num)), dtype=np.bool)
        hf.attrs.create('sample_rate', data=sample_rate, dtype=np.int32)

        # Forward data to a model in mini-batches
        for n, batch_data_dict in enumerate(generator):
            print(n, 'output_int16: ', output_int16)
            batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)

            with torch.no_grad():
                model.eval()
                batch_output = model(batch_waveform)

            # print(batch_output['stft'].size())
            # print(batch_output['logmel'].size())
            # # torch.Size([32, 1, 1001, 513])
            # # torch.Size([32, 1, 1001, 64])

            stft = batch_output['stft'].data.cpu().numpy()[:, 0, :, :]
            logmel = batch_output['logmel'].data.cpu().numpy()[:, 0, :, :]
            # print(stft.shape)
            # print(logmel.shape)
            # (32, 1001, 513)
            # (32, 1001, 64)

            # print(batch_data_dict['audio_name'])
            # print(batch_data_dict['audio_name'].shape)
            # #  'Y-2X03mO3T_U.wav' 'Y-2hQKCE-oTI.wav' 'Y-30H9V1IKps.wav'
            # #  'Y-38Qgsbh7NQ.wav' 'Y-3HYdaJyF4U.wav' 'Y-3IGxVTJvgI.wav' ''
            # #  'Y-3hKkjKmIGE.wav' 'Y-3pPrlCm6gg.wav']
            # # (32,)

            for i in range(len(stft)):
                sample_index = n * len(stft) + i
                # print('now: ', i)
                # print(logmel[i].shape, hf['logmel'][sample_index].shape)  # (1001, 64) (1001, 64)
                # print(batch_data_dict['target'][i].shape, hf['target'][sample_index].shape)
                # (527,) (527,)
                hf['audio_name'][sample_index] = str(batch_data_dict['audio_name'][i]).encode()

                # print(logmel[i].dtype)
                # float32

                if output_int16:
                    hf['logmel'][sample_index] = logmel[i].astype(np.int16)  # (1001, 64)
                    hf['stft'][sample_index] = stft[i].astype(np.int16)

                    # 请注意，将 float32 类型的数组转换为 int16 类型的数组时，数据可能会发生截断或舍入，从而导致数据的损失。
                    # 因此，在进行此类转换时，请仔细检查您的数据并确保不会丢失重要信息。
                else:
                    hf['logmel'][sample_index] = logmel[i]  # (1001, 64)
                    hf['stft'][sample_index] = stft[i]

                hf['target'][sample_index] = (batch_data_dict['target'][i]).astype(np.bool)

                # print(hf['audio_name'][sample_index])
                # print(hf['logmel'][sample_index].shape)
                # # b'Y--PJHxphWEs.wav'
                # # (1001, 64)

            append_to_dict(output_dict, 'audio_name', batch_data_dict['audio_name'])

            append_to_dict(output_dict, 'clipwise_output',
                           batch_output['clipwise_output'].data.cpu().numpy())

            if 'segmentwise_output' in batch_output.keys():
                append_to_dict(output_dict, 'segmentwise_output',
                               batch_output['segmentwise_output'].data.cpu().numpy())

            if 'framewise_output' in batch_output.keys():
                append_to_dict(output_dict, 'framewise_output',
                               batch_output['framewise_output'].data.cpu().numpy())

            if return_input:
                append_to_dict(output_dict, 'waveform', batch_data_dict['waveform'])

            if return_target:
                if 'target' in batch_data_dict.keys():
                    append_to_dict(output_dict, 'target', batch_data_dict['target'])

            if n % 10 == 0:
                print(' --- Inference time: {:.3f} s / 10 iterations ---'.format(
                    time.time() - time1))
                time1 = time.time()

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict



def forward_hyb_output_logmel(model, generator, audios_num, sample_rate, classes_num,
                                    waveforms_hdf5_path,
                                   return_input=False, return_target=False):
    output_dict = {}
    device = next(model.parameters()).device
    time1 = time.time()

    frame_num = 1001
    mel_bins = 64

    if sample_rate==16000:
        stft_bins = 257
    elif sample_rate==32000:
        stft_bins = 513

    print('save to: ', waveforms_hdf5_path)
    with h5py.File(waveforms_hdf5_path, 'w') as hf:
        hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S20')
        hf.create_dataset('logmel', shape=((audios_num, frame_num, mel_bins)), dtype=np.float32)
        # hf.create_dataset('stft', shape=((audios_num, frame_num, stft_bins)), dtype=np.float32)
        hf.create_dataset('target', shape=((audios_num, classes_num)), dtype=np.bool)
        hf.attrs.create('sample_rate', data=sample_rate, dtype=np.int32)

        # Forward data to a model in mini-batches
        for n, batch_data_dict in enumerate(generator):
            print(n,)

            batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)

            with torch.no_grad():
                model.eval()
                batch_output = model(batch_waveform)

            # print(batch_output['stft'].size())
            # print(batch_output['logmel'].size())
            # # torch.Size([32, 1, 1001, 513])
            # # torch.Size([32, 1, 1001, 64])

            # stft = batch_output['stft'].data.cpu().numpy()[:, 0, :, :]
            logmel = batch_output['logmel'].data.cpu().numpy()[:, 0, :, :]
            # print(stft.shape)
            # print(logmel.shape)
            # (32, 1001, 513)
            # (32, 1001, 64)

            # print(batch_data_dict['audio_name'])
            # print(batch_data_dict['audio_name'].shape)
            # #  'Y-2X03mO3T_U.wav' 'Y-2hQKCE-oTI.wav' 'Y-30H9V1IKps.wav'
            # #  'Y-38Qgsbh7NQ.wav' 'Y-3HYdaJyF4U.wav' 'Y-3IGxVTJvgI.wav' ''
            # #  'Y-3hKkjKmIGE.wav' 'Y-3pPrlCm6gg.wav']
            # # (32,)

            for i in range(len(logmel)):
                sample_index = n * len(logmel) + i
                # print('now: ', i)
                # print(logmel[i].shape, hf['logmel'][sample_index].shape)  # (1001, 64) (1001, 64)
                # print(batch_data_dict['target'][i].shape, hf['target'][sample_index].shape)
                # (527,) (527,)
                hf['audio_name'][sample_index] = str(batch_data_dict['audio_name'][i]).encode()

                # print(logmel[i].dtype)
                # float32

                hf['logmel'][sample_index] = logmel[i]  # (1001, 64)

                hf['target'][sample_index] = (batch_data_dict['target'][i]).astype(np.bool)

                # print(hf['audio_name'][sample_index])
                # print(hf['logmel'][sample_index].shape)
                # # b'Y--PJHxphWEs.wav'
                # # (1001, 64)

            append_to_dict(output_dict, 'audio_name', batch_data_dict['audio_name'])

            append_to_dict(output_dict, 'clipwise_output',
                           batch_output['clipwise_output'].data.cpu().numpy())

            if 'segmentwise_output' in batch_output.keys():
                append_to_dict(output_dict, 'segmentwise_output',
                               batch_output['segmentwise_output'].data.cpu().numpy())

            if 'framewise_output' in batch_output.keys():
                append_to_dict(output_dict, 'framewise_output',
                               batch_output['framewise_output'].data.cpu().numpy())

            if return_input:
                append_to_dict(output_dict, 'waveform', batch_data_dict['waveform'])

            if return_target:
                if 'target' in batch_data_dict.keys():
                    append_to_dict(output_dict, 'target', batch_data_dict['target'])

            if n % 10 == 0:
                print(' --- Inference time: {:.3f} s / 10 iterations ---'.format(
                    time.time() - time1))
                time1 = time.time()

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict


def forward_full_train_output_logmel_emb2048(model, generator, audios_num, sample_rate, classes_num,
                              waveforms_hdf5_path, emb2048_hdf5_path):
    device = next(model.parameters()).device

    frame_num = 1001
    mel_bins = 64

    if sample_rate == 16000:
        stft_bins = 257
    elif sample_rate == 32000:
        stft_bins = 513

    #
    # Forward data to a model in mini-batches
    logmel_list = []
    audio_name_list = []
    target_list = []
    emb2048_list = []
    # for n, batch_data_dict in enumerate(generator):
    #     print(n, )
    #
    #     # if n > 3:
    #     #     break
    #
    #     batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)
    #
    #     with torch.no_grad():
    #         model.eval()
    #         batch_output = model(batch_waveform)
    #
    #     # stft = batch_output['stft'].data.cpu().numpy()[:, 0, :, :]
    #     logmel = batch_output['logmel'].data.cpu().numpy()[:, 0, :, :]
    #     logmel_list.append(logmel)
    #     audio_name_list.append(batch_data_dict['audio_name'])
    #     target_list.append(batch_data_dict['target'])
    #
    #     embedding = batch_output['embedding'].data.cpu().numpy()
    #     emb2048_list.append(embedding)
    #
    # logmel_list = np.concatenate(logmel_list)
    # audio_name_list = np.concatenate(audio_name_list)
    # target_list = np.concatenate(target_list)
    # emb2048_list = np.concatenate(emb2048_list)
    #
    # print(logmel_list.shape, audio_name_list.shape, target_list.shape, emb2048_list.shape)
    # # (128, 1001, 64) (128,) (128, 527) (128, 2048)
    # # print(logmel_list[0], audio_name_list[0], target_list[0], emb2048_list[0])

    print('save to: ', waveforms_hdf5_path)
    with h5py.File(waveforms_hdf5_path, 'w') as hf:

        for n, batch_data_dict in enumerate(generator):
            print(n, )

            # if n > 3:
            #     break

            batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)

            with torch.no_grad():
                model.eval()
                batch_output = model(batch_waveform)

            # stft = batch_output['stft'].data.cpu().numpy()[:, 0, :, :]
            logmel = batch_output['logmel'].data.cpu().numpy()[:, 0, :, :]
            logmel_list.append(logmel)
            audio_name_list.append(batch_data_dict['audio_name'])
            target_list.append(batch_data_dict['target'])

            embedding = batch_output['embedding'].data.cpu().numpy()
            emb2048_list.append(embedding)

        logmel_list = np.concatenate(logmel_list)
        audio_name_list = np.concatenate(audio_name_list)
        target_list = np.concatenate(target_list)
        emb2048_list = np.concatenate(emb2048_list)

        print(logmel_list.shape, audio_name_list.shape, target_list.shape, emb2048_list.shape)
        # (128, 1001, 64) (128,) (128, 527) (128, 2048)
        # print(logmel_list[0], audio_name_list[0], target_list[0], emb2048_list[0])

        audios_num = len(logmel_list)
        hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S20')
        hf.create_dataset('logmel', shape=((audios_num, frame_num, mel_bins)), dtype=np.float32)
        # hf.create_dataset('stft', shape=((audios_num, frame_num, stft_bins)), dtype=np.float32)
        hf.create_dataset('target', shape=((audios_num, classes_num)), dtype=np.bool)
        hf.attrs.create('sample_rate', data=sample_rate, dtype=np.int32)

        # Forward data to a model in mini-batches
        for sample_index, (logmel, audio_name, target) in enumerate(zip(logmel_list, audio_name_list, target_list)):
            hf['audio_name'][sample_index] = str(audio_name).encode()
            hf['logmel'][sample_index] = logmel   # (1001, 64)
            hf['target'][sample_index] = target.astype(np.bool)
    #
    print('save to: ', emb2048_hdf5_path)
    with h5py.File(emb2048_hdf5_path, 'w') as hf:
        hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S20')
        hf.create_dataset('emb2048', shape=((audios_num, 2048)), dtype=np.float32)
        hf.create_dataset('target', shape=((audios_num, classes_num)), dtype=np.bool)
        hf.attrs.create('sample_rate', data=sample_rate, dtype=np.int32)

        # Forward data to a model in mini-batches
        for sample_index, (emb2048, audio_name, target) in enumerate(zip(emb2048_list,
                                                                        audio_name_list,
                                                                        target_list)):
            hf['audio_name'][sample_index] = str(audio_name).encode()
            hf['emb2048'][sample_index] = emb2048  # (1001, 64)
            hf['target'][sample_index] = target.astype(np.bool)


def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the 
    resolution reduction in downsampling of a CNN.
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate

    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value 
    is the same as the value of the last frame.

    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad

    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1 :, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, audio_length):
    """Count flops. Code modified from others' implementation.
    """
    multiply_adds = True
    list_conv2d=[]
    def conv2d_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
 
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
 
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
 
        list_conv2d.append(flops)

    list_conv1d=[]
    def conv1d_hook(self, input, output):
        batch_size, input_channels, input_length = input[0].size()
        output_channels, output_length = output[0].size()
 
        kernel_ops = self.kernel_size[0] * (self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
 
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_length
 
        list_conv1d.append(flops)
 
    list_linear=[] 
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
 
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()
 
        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)
 
    list_bn=[] 
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)
 
    list_relu=[] 
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement() * 2)
 
    list_pooling2d=[]
    def pooling2d_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
 
        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
 
        list_pooling2d.append(flops)

    list_pooling1d=[]
    def pooling1d_hook(self, input, output):
        batch_size, input_channels, input_length = input[0].size()
        output_channels, output_length = output[0].size()
 
        kernel_ops = self.kernel_size[0]
        bias_ops = 0
        
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_length
 
        list_pooling2d.append(flops)
 
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, nn.Conv2d):
                net.register_forward_hook(conv2d_hook)
            elif isinstance(net, nn.Conv1d):
                net.register_forward_hook(conv1d_hook)
            elif isinstance(net, nn.Linear):
                net.register_forward_hook(linear_hook)
            elif isinstance(net, nn.BatchNorm2d) or isinstance(net, nn.BatchNorm1d):
                net.register_forward_hook(bn_hook)
            elif isinstance(net, nn.ReLU):
                net.register_forward_hook(relu_hook)
            elif isinstance(net, nn.AvgPool2d) or isinstance(net, nn.MaxPool2d):
                net.register_forward_hook(pooling2d_hook)
            elif isinstance(net, nn.AvgPool1d) or isinstance(net, nn.MaxPool1d):
                net.register_forward_hook(pooling1d_hook)
            else:
                print('Warning: flop of module {} is not counted!'.format(net))
            return
        for c in childrens:
            foo(c)

    # Register hook
    foo(model)
    
    device = device = next(model.parameters()).device
    input = torch.rand(1, audio_length).to(device)

    out = model(input)
 
    total_flops = sum(list_conv2d) + sum(list_conv1d) + sum(list_linear) + \
        sum(list_bn) + sum(list_relu) + sum(list_pooling2d) + sum(list_pooling1d)
    
    return total_flops