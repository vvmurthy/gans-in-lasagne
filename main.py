import sys
from config import config


def run_function(mode='train'):
    configuration = config()
    if mode == 'train':
        configuration['train_function'](configuration)
    elif mode == 'test':
        configuration['test_function'](configuration)

kwargs = {}
if len(sys.argv) > 1:
    kwargs['mode'] = sys.argv[1]

run_function(**kwargs)
