import sys
from config import config


def run_function(mode='train'):
    model = config()
    if mode == 'train':
        model.train()
    elif mode == 'test':
        model.test()

kwargs = {}
if len(sys.argv) > 1:
    kwargs['mode'] = sys.argv[1]

run_function(**kwargs)
