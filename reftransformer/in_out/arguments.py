"""
Handle arguments for train/test scripts.

The MIT License (MIT)
Originally created at 5/25/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Ahmed (@gmail.com)
"""

import argparse
import json
import pprint
import os.path as osp
from shutil import copyfile
from datetime import datetime
from argparse import ArgumentParser
from enum import Enum, unique

from ..utils import str2bool, create_dir


def parse_arguments(notebook_options=None):
    """Parse the arguments for the training (or test) execution of a ReferIt3D net.
    :param notebook_options: (list) e.g., ['--max-distractors', '100'] to give/parse arguments from inside a jupyter notebook.
    :return:
    """
    parser = argparse.ArgumentParser(description='ReferIt3D Nets + Ablations')
    parser.add_argument('--config-file', type=str, default=None, help='config file')
    parser.add_argument('--mode', type=str, default='train', help='train or evaluate')
    args = parser.parse_args()

    assert args.config_file is not None

    with open(args.config_file, 'r') as fin:
        configs_dict = json.load(fin)
        configs_dict['mode'] = args.mode
        apply_configs(args, configs_dict)

    # Create logging related folders and arguments
    if args.log_dir:
        timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        args.log_dir = osp.join(args.log_dir, timestamp)

        args.checkpoint_dir = create_dir(osp.join(args.log_dir, 'checkpoints'))
        args.tensorboard_dir = create_dir(osp.join(args.log_dir, 'tb_logs'))

    if args.resume_path and not args.log_dir:  # resume and continue training in previous log-dir.
        checkpoint_dir = osp.split(args.resume_path)[0]  # '/xxx/yyy/log_dir/checkpoints/model.pth'
        args.checkpoint_dir = checkpoint_dir
        args.log_dir = osp.split(checkpoint_dir)[0]
        args.tensorboard_dir = osp.join(args.log_dir, 'tb_logs')

        # Copy the arguments python file to the log
        copyfile('arguments.py', osp.join(args.log_dir, 'arguments.py'))

    # Print them nicely
    args_dict = vars(args)
    args_string = pprint.pformat(args_dict)
    print(args_string)

    out = osp.join(args.log_dir, 'config.json.txt')
    with open(out, 'w') as fout:
        fout.write(args_string)

    return args


def read_saved_args(config_file, override_args=None, verbose=True):
    """
    :param config_file:
    :param override_args: dict e.g., {'gpu': '0'}
    :param verbose:
    :return:
    """
    parser = ArgumentParser()
    args = parser.parse_args([])
    with open(config_file, 'r') as f_in:
        args.__dict__ = json.load(f_in)

    if override_args is not None:
        for key, val in override_args.items():
            args.__setattr__(key, val)

    if verbose:
        args_string = pprint.pformat(vars(args))
        print(args_string)

    return args


def apply_configs(args, config_dict):
    for k, v in config_dict.items():
        setattr(args, k, v)
