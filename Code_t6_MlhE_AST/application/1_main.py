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
    training_label_type = 'hard'
    label_type = '_' + training_label_type

    basic_name = 'system'


    suffix, system_name = define_system_name(basic_name=basic_name)
    system_path = os.path.join(os.getcwd(), system_name)

    Model = MlhE_AST
    classes_num = len(config.labels)
    event_class = len(config.event_labels)
    model = Model(scene_dim=classes_num, event_dim=event_class)

    models_dir = system_path + label_type

    print(model)

    if config.cuda:
        model.cuda()

    # Data generator
    hdf5_path = os.path.join(config.dataset_dir, 'AST', 'ast_D2018T1A_training.h5')
    generator = DataGenerator_ast_scene_event_add_onthot_scene(hdf5_path,
                                                               training_label_type,
                                                               label_type,
                                                               normalization=True)

    training = training_process_accumulating_gradients
    training(generator, model, config.cuda, models_dir, training_label_type,)
    print('Training is done!!!')



if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















