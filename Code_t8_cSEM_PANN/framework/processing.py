import framework.config as config
import time, os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from framework.utilities import create_folder, calculate_accuracy, calculate_confusion_matrix, print_accuracy, \
    plot_confusion_matrix, plot_confusion_matrix_each_file
from framework.models_pytorch import move_data_to_gpu
from sklearn import metrics


def define_system_name(alpha=None, basic_name='system', att_dim=None, n_heads=None,
                       batch_size=config.batch_size, epochs=config.epochs):
    suffix = ''
    if alpha:
        suffix = suffix.join([str(each) for each in alpha]).replace('.', '')

    sys_name = basic_name
    sys_suffix = '_b' + str(batch_size) + '_e' + str(epochs) \
                 + '_attd' + str(att_dim) + '_h' + str(n_heads) if att_dim is not None and n_heads is not None \
        else '_b' + str(batch_size)  + '_e' + str(epochs)

    sys_suffix = sys_suffix + '_cuda' + str(config.cuda_seed) if config.cuda_seed is not None else sys_suffix
    system_name = sys_name + sys_suffix if sys_suffix is not None else sys_name

    return suffix, system_name



def forward(model, generate_func, cuda, return_target):
    """Forward data to a model.

    Args:
      generate_func: generate function
      cuda: bool
      return_target: bool

    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """

    outputs = []
    audio_names = []

    if return_target:
        targets = []

    # Evaluate on mini-batch
    for data in generate_func:

        if return_target:
            (batch_x, batch_y, batch_audio_names) = data

        else:
            (batch_x, batch_audio_names) = data

        batch_x = move_data_to_gpu(batch_x, cuda)



        model.eval()
        with torch.no_grad():
            batch_output = model(batch_x)  # torch.Size([16, 10])
            # batch_output = F.softmax(batch_output, dim=-1)

        # Append data
        outputs.append(batch_output.data.cpu().numpy())
        audio_names.append(batch_audio_names)

        if return_target:
            targets.append(batch_y)

    dict = {}

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_name'] = audio_names

    if return_target:
        targets = np.concatenate(targets, axis=0)
        dict['target'] = targets

    return dict


def forward_asc_aec(model, generate_func, cuda, return_target):
    outputs = []
    outputs_event = []
    audio_names = []

    if return_target:
        targets = []
        targets_event = []

    no_event = True
    # Evaluate on mini-batch
    for data in generate_func:
        if return_target:
            if len(data) == 4:
                (batch_x, batch_y, batch_y_event, batch_audio_names) = data
                no_event = False
            elif len(data) == 3:
                (batch_x, batch_y, batch_audio_names) = data

        else:
            if len(data) == 3:
                (batch_x, batch_y_event, batch_audio_names) = data
                no_event = False
            elif len(data) == 2:
                (batch_x, batch_audio_names) = data

        batch_x = move_data_to_gpu(batch_x, cuda)
        # print(batch_x.size())

        model.eval()
        with torch.no_grad():
            all_output = model(batch_x)  # torch.Size([16, 10])
            batch_output, batch_output_event = all_output[0], all_output[1]
            batch_output = F.softmax(batch_output, dim=-1)
            outputs.append(batch_output.data.cpu().numpy())
            outputs_event.append(batch_output_event.data.cpu().numpy())

        audio_names.append(batch_audio_names)

        if return_target:
            targets.append(batch_y)
            if not no_event:
                targets_event.append(batch_y_event)

    dict = {}

    if len(outputs):
        outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    if len(outputs_event):
        outputs_event = np.concatenate(outputs_event, axis=0)
    dict['outputs_event'] = outputs_event

    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_name'] = audio_names

    if return_target:
        targets = np.concatenate(targets, axis=0)
        dict['target'] = targets
        if len(targets_event):
            targets_event = np.concatenate(targets_event, axis=0)
            dict['targets_event'] = targets_event

    return dict



def evaluate_asc_aec(model, generator, data_type, devices, max_iteration, cuda):
    """Evaluate

    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      devices: list of devices, e.g. ['a'] | ['a', 'b', 'c']
      max_iteration: int, maximum iteration for validation
      cuda: bool.

    Returns:
      accuracy: float
    """

    # Generate function
    generate_func = generator.generate_validate(data_type=data_type,
                                                devices=devices,
                                                shuffle=True,
                                                max_iteration=max_iteration)

    # Forward
    dict = forward_asc_aec(model=model,
                          generate_func=generate_func,
                          cuda=cuda,
                          return_target=True)

    outputs = dict['output']  # (audios_num, classes_num)
    targets = dict['target']  # (audios_num, classes_num)
    predictions = np.argmax(outputs, axis=-1)  # (audios_num,)
    classes_num = outputs.shape[-1]
    accuracy = calculate_accuracy(targets, predictions, classes_num, average='macro')

    outputs_event = dict['outputs_event']  # (audios_num, classes_num)
    targets_event = dict['targets_event']  # (audios_num, classes_num)
    # print('targets_event.shape: ', targets_event.shape)

    aucs = []
    for i in range(targets_event.shape[0]):
        test_y_auc, pred_auc = targets_event[i, :], outputs_event[i, :]
        if np.sum(test_y_auc):
            test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
            aucs.append(test_auc)
    final_auc = sum(aucs) / len(aucs)
    # print('Auc:', final_auc)

    return accuracy, final_auc



def training_process_cSEM(generator, model, cuda, models_dir,
                                      training_label_type, epochs=config.epochs,
                                      batch_size=config.batch_size,  lr_init = config.lr_init,):
    create_folder(models_dir)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08)

    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()
    if training_label_type == 'soft':
        event_loss = torch.nn.MSELoss()
    elif training_label_type == 'hard':
        event_loss = torch.nn.BCELoss()

    sample_num = len(generator.train_audio_indexes)
    val_sample_num = len(generator.validate_audio_indexes)
    one_epoch = int(sample_num / batch_size)
    print('one_epoch: ', one_epoch, 'is 1 epoch')
    print('really batch size: ', batch_size)
    check_iter = int(one_epoch)
    print('  validating every: ', check_iter, ' iteration')

    val_acc_scene = []
    val_auc_event = []

    # Train on mini batches
    for iteration, all_data in enumerate(generator.generate_train()):

        train_bgn_time = time.time()

        if len(all_data) == 2:
            (batch_x, batch_y) = all_data
        elif len(all_data) == 3:
            (batch_x, batch_y, batch_y_event) = all_data

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)
        batch_y_event = move_data_to_gpu(batch_y_event, cuda)

        model.train()

        optimizer.zero_grad()

        x_scene_linear, x_event_linear, inferred_scene, inferred_event = model(batch_x)

        x_event_sigmoid = F.sigmoid(x_event_linear)
        loss_event = event_loss(x_event_sigmoid, batch_y_event)

        x_scene_softmax = F.log_softmax(x_scene_linear, dim=-1)
        loss_scene = F.nll_loss(x_scene_softmax, batch_y)

        ############################################################################################################
        loss_infer_scene = mse_loss(inferred_scene, x_scene_softmax)
        loss_infer_event = mse_loss(inferred_event, x_event_sigmoid)

        optimizer.zero_grad()

        loss_common = loss_scene + loss_infer_scene + loss_event + loss_infer_event

        loss_common.backward()
        optimizer.step()

        if iteration % check_iter == 0 and iteration > 0:
            train_fin_time = time.time()
            va_acc, va_event_auc = evaluate_asc_aec(model=model,
                                                    generator=generator,
                                                    data_type='validate',
                                                    devices=config.devices,
                                                    max_iteration=None,
                                                    cuda=cuda)

            val_acc_scene.append(va_acc)
            val_auc_event.append(va_event_auc)

            print('E: ', '%.4f' % (iteration/one_epoch), ' val_scene_acc: %.3f' % va_acc,
                  ' val_event_auc: %.3f' % va_event_auc)

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            print('epoch: {}, train time: {:.3f} s, iteration time: {:.3f} ms, validate time: {:.3f} s,  '
                  'inference time : {:.3f} ms'
                  .format('%.2f' % (iteration / one_epoch), train_time, (train_time / sample_num) * 1000, validate_time,
                          1000 * validate_time / val_sample_num))

        # # Reduce learning rate
        # check_itera_step = int(itera_step * accumulation_steps)
        # if lr_decay and (iteration % check_itera_step == 0 > 0):
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.9

        # Stop learning
        if iteration > (epochs * one_epoch):
            save_out_dict = {'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(),}
            save_out_path = os.path.join(models_dir,
                                         'final_model' + config.endswith)
            torch.save(save_out_dict, save_out_path)
            print('Model saved to {}'.format(save_out_path))


            final_test = 1
            if final_test:
                va_acc, va_event_auc = evaluate_asc_aec(model=model,
                                                        generator=generator,
                                                        data_type='validate',
                                                        max_iteration=None,
                                                        cuda=cuda,
                                                        training_label_type=training_label_type)
                val_acc_scene.append(va_acc)
                val_auc_event.append(va_event_auc)

                print('Training is done!!!')

                print('Fianl: ', '%.4f' % (iteration/one_epoch),
                      ' val_scene_acc: %.6f' % va_acc,
                      ' val_event_auc: %.6f' % va_event_auc)

            break























