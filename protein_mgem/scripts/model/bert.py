import argparse
import os
from pathlib import Path

import yaml
from tape.training import run_train

# Define learning tasks
from scripts import tape_elements

os.environ['NUMEXPR_MAX_THREADS'] = '16'


def main():
    args = get_args()
    print('#' * 80)
    print(args.hp)
    print('#' * 80)

    if not args.tissue:
        data_dir = str(args.res_dir / 'data')
    else:
        data_dir = str(args.res_dir / 'per_tissue_config' / ('data_' + args.tissue))

    run_train(
        model_type = tape_elements.tape_model_name,
        task = args.task,
        save_freq = get_save_freq(args),
        learning_rate = float(args.hp['learning_rate']),
        num_train_epochs = int(args.hp['num_train_epochs']),
        data_dir = data_dir,
        output_dir = str(args.res_dir / 'bert'),
        log_dir = str(args.res_dir / 'logs'),
        batch_size = int(args.hp['batch_size']),
        gradient_accumulation_steps = int(args.hp['gradient_accumulation_steps']),
        model_config_file = str(args.config),
        patience = int(args.hp['patience'])
    )


def get_save_freq(args):
    if 'save_freq' in args.hp:
        return int(args.hp['save_freq'])
    else:
        return 'improvement'


def get_args():
    parser = argparse.ArgumentParser(description="BERT")
    parser.add_argument('--res_dir',
                        required=True,
                        type=str,
                        help='Results dir to save models')
    parser.add_argument('--config',
                        required=True,
                        type=str,
                        help='JSON config (arch + hyperparams) file')
    parser.add_argument('--task',
                        required=False,
                        type=str,
                        default='learn_abundance',
                        help='TAPE task to run')
    parser.add_argument('--tissue',
                        required=False,
                        type=str,
                        default='',
                        help='Tissue to learn on')
    args = parser.parse_args()
    args.res_dir = Path(args.res_dir)
    with open(args.config, 'r') as config_file:
        args.hp = yaml.load(config_file, Loader=yaml.FullLoader)
    return args


if __name__ == '__main__':
    main()
