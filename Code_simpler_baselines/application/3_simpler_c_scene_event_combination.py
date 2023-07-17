import sys, os, argparse

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from framework.utilities import *
from framework.models_pytorch import *
from framework.data_generator import *
from framework.processing import *




def main(argv):
    label_type = ''
    cuda = config.cuda
    batch_size = 64
    epochs = 100

    basic_name = 'system'

    suffix, system_name = define_system_name(basic_name=basic_name)
    system_path = os.path.join(os.getcwd(), system_name)

    Model = Simpler_baseline_scene_event_combination

    models_dir = system_path + label_type

    classes_num = len(config.labels)
    event_class = 527
    model = Model(classes_num, event_class, batchnormal=True)

    print(model)

    if config.cuda:
        model.cuda()

    # Data generator
    hdf5_file = os.path.join(config.dataset_dir, 'PANN', 'kong_mel', 'development.h5')
    print('using: ', hdf5_file)
    generator = DataGenerator_scene_event(hdf5_path=hdf5_file, normalization=True)

    training = training_process_model_C
    training(generator, model, cuda, models_dir, epochs, batch_size,  lr_init=config.lr_init)
    print('Training is done!!!')



if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















