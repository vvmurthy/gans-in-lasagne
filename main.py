import os
import sys

from src.models.began import BEGAN
from src.datasets.celeba import CelebA
from src.datasets.mnist import Mnist
from src.models.icgan import IcGAN


def config(**kwargs):
    configuration = {}

    # Generate configuration for use in training / testing
    # configuration creates a dict object with hyperparameter names as keys
    # Hyperparameters:
    #   - im_dir = directory where images + label file is located
    #   - folder_name = folder to store models + images in
    #   - dataset = which dataset to run (choices are 'mnist' and 'celeba')
    #   - file_loader = for moving images to memory, which file loader to use
    #   - model = model to train ('icgan' or 'BEGAN')

    configuration['im_dir'] = kwargs.get('im_dir', os.getcwd() + '/mnist/')
    folder_name = kwargs.get('folder_name', 'began_celeba2')
    dataset = kwargs.get('dataset', 'mnist')
    configuration['model'] = kwargs.get('model', 'icgan')

    # Check dataset + combinations are supported
    assert (dataset in ['celeba', 'mnist']), "Dataset not supported"

    # Check model is supported
    assert (configuration['model'] in ['icgan', 'began']), "Model not implemented"

    # Configure some settings based on dataset choice
    # Then Initialize Dataset
    print("Initializing Dataset...")
    if 'mnist' == dataset:
        dataset = Mnist(configuration['im_dir'], **configuration)
    elif 'celeba' == dataset:
        dataset = CelebA(configuration['im_dir'], **configuration)

    # Build models
    if configuration['model'] == 'began':
        model = BEGAN(dataset, folder_name, **configuration)
    elif configuration['model'] == 'icgan':
        model = IcGAN(dataset, folder_name, **configuration)

    # Save config dictionary 
    try:
        import pickle
        filename = os.getcwd() +  '/' + folder_name + '/stats/config.pkl'
        pickle.dump(configuration, open(filename, 'wb'), 
                    protocol=pickle.HIGHEST_PROTOCOL)
    except ImportError:
        pass

    print('Done')

    return model


def run_function(mode, **kwargs):
    model = config()
    if mode == 'train':
        model.train()
    elif mode == 'test':
        model.test()


kwargs = {}
if len(sys.argv) > 1:
    mode = sys.argv[1]
if len(sys.argv) > 2:
    kwargs['im_dir'] = sys.argv[2]
if len(sys.argv) > 3:
    kwargs['folder_name'] = sys.argv[3]
if len(sys.argv) > 4:
    kwargs['dataset'] = sys.argv[4]
if len(sys.argv) > 5:
    kwargs['model'] = sys.argv[5]

run_function(mode, **kwargs)
