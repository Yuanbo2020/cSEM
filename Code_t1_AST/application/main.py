import sys, os, argparse

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from framework.utilities import *
from framework.models_pytorch import *
from framework.data_generator import *
from framework.processing import *


def main(argv):
    feature_type = '_astf'
    norm_type = '_astn'
    label_type = feature_type + norm_type

    if '_astn' in norm_type:
        ast_normalize()

    basic_name = 'system'


    suffix, system_name = define_system_name(basic_name=basic_name)
    system_path = os.path.join(os.getcwd(), system_name)

    Model = AST

    models_dir = system_path + label_type


    classes_num = len(config.labels)
    model = Model(label_dim=classes_num)

    print(model)

    if config.cuda:
        model.cuda()

    # Data generator
    hdf5_path = os.path.join(config.dataset_dir, 'AST', 'ast_D2018T1A_training.h5')
    generator = DataGenerator_ast_scene(hdf5_path=hdf5_path)

    training = training_process
    training(generator, model, config.cuda, models_dir, label_type=label_type)
    print('Training is done!!!')



if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















