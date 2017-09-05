import os

from models.began import BEGAN

from src.datasets import CelebA
from src.datasets import Mnist
from src.models.icgan import IcGAN


def init_config():
    configuration = {}

    # Generate configuration for use in training / testing
    # configuration creates a dict object with hyperparameter names as keys
    # Hyperparameters:
    #   - bz = batchsize
    #   - li = length of image (assumed to be square)
    #   - images_in_mem = how many images to store in memory at once during training
    #     (set to None for all images)
    #   - nc = number of channels in original image (3 for celebA, 1 for MNIST)
    #   - lab_ln = number of labels in dataset
    #   - lr = learning rate (Perarnau et al use 0.0002)
    #   - num_epochs = number of epochs (full passes over training set) to perform
    #   - im_dir = directory where images + label file is located
    #   - folder_name = folder to store models + images in
    #   - dataset = which dataset to run (choices are 'mnist' and 'celeba')
    #   - file_loader = for moving images to memory, which file loader to use
    #   - model = model to train (ICGAN or BEGAN)
    #   - dataset_loader = internal - which function that will be used to load a batch of a dataset
    #   - batch_iterator = internal - batch iterators are specific to conditional / unconditional training
    #   - train_function = function to train specified model
    #   - test_function = function to test specified model
    ##### Hyperparameters for BEGAN only
    #   - gamma = the hyperparameter gamma specified in Berthelot et al
    #   - num_filters = the number of filters to use (N in Berthelot et al)
    #   - k_t = initial value for k_t in Berthelot et al (they use 0)
    #   - offset = Berthelot et al train by linearly increasing number of filters
    #   after strided convolution or nearest neighbor resize. This offsets
    #   the start of linearly increasing number of filters for modest hardware
    #   I use 5 ( no increase in filters) because of hardware limitations
    #   - num_hidden = the length of variable z (input to generator)
    #   - z_var = the method of generating z (so far there is only support for 1 method)

    configuration['im_dir'] = os.getcwd() + '/mnist/'
    folder_name = 'began_celeba2'
    dataset = 'mnist'
    configuration['model'] = 'icgan'

    return configuration, folder_name, dataset


def config():
    # Initialize configuration
    configuration, folder_name, dataset = init_config()

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
