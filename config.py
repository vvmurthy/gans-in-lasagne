import os
from math import sqrt
import datasets.celeba as celeba
import datasets.mnist as mnist
from train.train_began import  train_began
from train.train_icgan import train_icgan
from test.test_icgan import test_icgan
from test.test_began import test_began
from utils.icgan_utils import modify_y, modify_y_celeba, randomize_y
from utils.icgan_utils import randomize_y_celeba
from utils.z_vars import z_var_uniform


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

    configuration['bz'] = 64
    configuration['images_in_mem'] = 12000
    configuration['lr'] = 0.0002 # Perarnau et al use 0.0002
    configuration['num_epochs'] = 25
    configuration['im_dir'] = os.getcwd() + '/celeba/'
    configuration['folder_name'] = 'began_celeba'
    configuration['dataset'] = ['celeba']
    configuration['file_loader'] = None
    configuration['model'] = 'began'
    configuration['dataset_loader'] = None
    configuration['batch_iterator'] = None
    configuration['train_function'] = None
    configuration['test_function'] = None

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
    if configuration['model'] == 'began':
        configuration['gamma'] = 0.7
        configuration['num_filters'] = 64
        configuration['k_t'] = 0
        configuration['offset'] = 5
        configuration['train_function'] = train_began
        configuration['test_function'] = test_began
        configuration['num_hidden'] = 64
        configuration['z_var'] = z_var_uniform
    ##### Hyperparameters for ICGAN
    if configuration['model'] == 'icgan':
        configuration['train_function'] = train_icgan
        configuration['test_function'] = test_icgan
        configuration['num_hidden'] = 100
        configuration['z_var'] = z_var_uniform
        if 'mnist' in configuration['dataset']:
            configuration['randomize_y'] = randomize_y
            configuration['modify_y'] = modify_y
        elif 'celeba' or 'celeba-128' in configuration['dataset']:
            configuration['modify_y'] = modify_y_celeba
            configuration['randomize_y'] = randomize_y_celeba
            

    return configuration


def check_config(configuration):
    # Check image directory exists
    assert (os.path.isdir(configuration['im_dir'])), "Image directory does not exist"

    # Check dataset + combinations are supported
    assert ('celeba' in configuration['dataset'] or 'mnist' in configuration['dataset']
            or 'celeba-128' in configuration['dataset']), "Dataset not supported"
            
    # Check that if BEGAN is used, the number of hidden units is a square number
    if configuration['model'] == 'began':
        assert (int(sqrt(configuration['num_hidden'])) == sqrt(configuration['num_hidden'])
        ),"Z vector length for BEGAN must be square of integer"
            
    # Check folders exist, otherwise make them
    base = os.getcwd() + '/' + configuration['folder_name'] + '/'
    if not os.path.isdir(base):
        os.mkdir(base)
        os.mkdir(base + 'models/')
        os.mkdir(base + 'images/')
        os.mkdir(base + 'stats/')

    # Check model is supported
    assert(configuration['model'] == 'icgan' or configuration['model'] == 'began'), "Model not implemented"


def config():
    # Initialize configuration
    configuration = init_config()

    # Check that it is a valid configuration
    check_config(configuration)

    # Configure some settings based on dataset choice
    # Then Initialize Dataset
    print("Initializing Dataset...")
    if 'mnist' in configuration['dataset'] and len(configuration['dataset']) == 1:
        configuration['nc'] = 1
        configuration['lab_ln'] = 10
        configuration['li'] = 32
        configuration['dataset_loader'] = mnist.load_files
        configuration['dataset_init'] = mnist.load_xy

    elif 'celeba' in configuration['dataset'] and len(configuration['dataset']) == 1:
        configuration['nc'] = 3
        configuration['lab_ln'] = 18
        configuration['li'] = 64
        configuration['dataset_loader'] = celeba.load_files
        configuration['dataset_init'] = celeba.load_xy

    elif 'celeba-128' in configuration['dataset'] and len(configuration['dataset']) == 1:
        configuration['nc'] = 3
        configuration['lab_ln'] = 18
        configuration['li'] = 128
        configuration['dataset_loader'] = celeba.load_files
        configuration['dataset_init'] = celeba.load_xy

    # Save config dictionary 
    try:
        import pickle
        filename = os.getcwd() +  '/' + configuration['folder_name'] + '/stats/config.pkl'
        pickle.dump(configuration, open(filename, 'wb'), 
                    protocol=pickle.HIGHEST_PROTOCOL)
    except ImportError:
        pass


    configuration['X_files_train'], configuration['y_train'], configuration['X_files_val'], \
    configuration['y_val'], \
    configuration['X_files_test'], configuration['y_test'], configuration['labels'] = configuration['dataset_init'](configuration['im_dir'])

    # Configure images in memory if not already done
    if configuration['images_in_mem'] == None:
        configuration['images_in_mem'] = configuration['X_files_train'].shape[0]

    print('Done')

    return configuration
