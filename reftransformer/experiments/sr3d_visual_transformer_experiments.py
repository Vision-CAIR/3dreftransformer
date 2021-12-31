#
# This file creates a set of experiments to see what are the best hyperparameters for the visual transformer model
# It will be used with the LSTM language encoder cause this model obtained the best accuracy et
#
from itertools import product
from typing import List

from reftransformer.experiments.main_config import DEFAULT_CONFIG
import json
import os
import pandas as pd
import numpy as np


def compute_hpo_vals_idx(hpo_grid) -> List[List[int]]:
    grid_dim_sizes = [len(hpo_grid.get(p)) for p in sorted(hpo_grid.keys())]
    vals_idx = [list(range(n)) for n in grid_dim_sizes]
    experiments_vals_idx = list(product(*vals_idx))
    return experiments_vals_idx


visual_transformer_grid_search = {
    'num_layers': [4],
    'num_heads': [8],
    'dropout_in_ff': [0.3],
    'dropout_in_attn': [0.1],
    'hidden_dim': [128],
    'use_rezero': [True],
    'forward_expansion': [512],
    'rel_loss': [True, False],
    'neg_loss': [True, False],
    'augment_with_sr3d': ['../sr3d.csv', '../sr3d+.csv'],
    's_vs_n_weight': [0.4],
    'scannet_file': ['../scans_all.pkl']
}

if __name__ == '__main__':
    # Read the default config
    default_config = DEFAULT_CONFIG

    # Prepare the experiments and create an experiment codes for each one
    counter = 0
    commands = {}
    gpu = 0

    res = compute_hpo_vals_idx(visual_transformer_grid_search)
    print(res)

    if not os.path.isdir('visual_transformer_exp_sr3d'):
        os.mkdir('visual_transformer_exp_sr3d')

    r_col = list(visual_transformer_grid_search.keys()) + ['Train accuracy', 'Test accuracy']
    t = pd.DataFrame(data=np.zeros((16, 14)), columns=r_col)

    for i_cfg in res:
        cfg = dict(default_config)
        cfg.update(visual_transformer_grid_search)

        cfg['log_dir'] = 'visual_transformer_exp_sr3d_{}'.format(counter)
        cfg['checkpoint_dir'] = cfg['log_dir'] + '/checkpoints'
        cfg['tensorboard_dir'] = cfg['log_dir'] + '/tensorboard'
        cfg['type'] = 'visual_transformer'

        for j, k in enumerate(sorted(visual_transformer_grid_search.keys())):
            cfg[k] = cfg[k][i_cfg[j]]
            if k == 'use_rezero':
                t.iloc[counter][k] = 1 if cfg[k] else 0
            else:
                t.iloc[counter][k] = cfg[k]

        cfg['gpu'] = gpu
        # generate python file
        commands[
            counter] = 'CUDA_VISIBLE_DEVICES={} python train_reftransformer.py --config visual_transformer_exp_sr3d/{}.json'.format(
            gpu, counter)

        with open('visual_transformer_exp_sr3d/{}.json'.format(counter), 'w') as fout:
            json.dump(cfg, fout, sort_keys=True, indent=4)

        counter += 1
        gpu = (gpu + 1) % 4

    print('Total number of experiments for visual transformer is: {}'.format(counter))
    assert counter == len(res)


    with open('visual_transformer_exp_sr3d_command_look_up.json.txt', 'w') as fout:
        json.dump(commands, fout, sort_keys=True, indent=4)

    t.to_excel('visual_transformer_exp_sr3d_table.xlsx')