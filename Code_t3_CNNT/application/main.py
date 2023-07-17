import sys, os, argparse

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from framework.utilities import *
from framework.models_pytorch import *
from framework.data_generator import *
from framework.processing import *


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main(argv):
    label_type = ''

    batch_size = 64
    epochs = 100

    basic_name = 'system'

    if config.cuda_seed:
        np.random.seed(config.cuda_seed)
        torch.manual_seed(config.cuda_seed)  # 为CPU设置随机种子
        if config.cuda:
            torch.cuda.manual_seed(config.cuda_seed)  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(config.cuda_seed)  # 为所有GPU设置随机种子

    suffix, system_name = define_system_name(basic_name=basic_name)
    system_path = os.path.join(os.getcwd(), system_name)

    Model = CnnT

    models_dir = system_path + label_type

    classes_num = len(config.labels)
    model = Model(classes_num, batchnormal=True)

    print(model)

    if config.cuda:
        model.cuda()

    # Data generator
    hdf5_file = os.path.join(config.dataset_dir, 'PANN', 'kong_mel', 'development.h5')
    print('using: ', hdf5_file)
    generator = DataGenerator_scene_event_D2018(hdf5_path=hdf5_file,
                                          normalization=True,
                                          dev_train_csv=config.dev_train_csv,
                                          dev_validate_csv=config.dev_validate_csv)

    training = training_process
    training(generator, model, config.cuda, models_dir, label_type=label_type)
    print('Training is done!!!')



if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















