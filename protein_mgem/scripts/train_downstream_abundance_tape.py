"""
Script that wraps the tape-train-distributed routine:
https://github.com/songlab-cal/tape#training-a-downstream-model
"""
import argparse
import inspect
import logging
import sys
from pathlib import Path

from tape.main import run_train #_distributed
from tape.models.modeling_bert import ProteinBertForValuePrediction

_this_filename = inspect.getframeinfo(inspect.currentframe()).filename
_this_path = Path(_this_filename).parent.resolve()
sys.path.append(str(_this_path))

import tape_elements


def main():
    run_args = get_args()
    model = ProteinBertForValuePrediction.from_pretrained('bert-base')
    # Seemed quite complicated to pass the model directly so relying on loading from file.
    pretrained_path = run_args.result_dir
    print(pretrained_path)
    model.save_pretrained(pretrained_path)

    # args ultimately passed to tape.training.run_train()
    train_args = argparse.Namespace(
        model_type = 'transformer',
        task = 'learn_abundance',
        exp_name = '_'.join([pretrained_path, 'transformer']),
        from_pretrained = pretrained_path,
        learning_rate = 1e-4,
        num_train_epochs = 100,
        batch_size = 2,
        gradient_accumulation_steps = 1,
        warmup_steps = 10000,
        eval_freq = 1,
        save_freq = 10,
        patience = -1,
        seed = 42,
        data_dir = str(Path(run_args.result_dir) / 'data'),
        output_dir = run_args.result_dir,
        log_dir = str(Path(run_args.result_dir) / 'logs'),

        model_config_file = None,
        nproc_per_node = 2,
        fp16 = False,
        local_rank = -1,
        debug = False,
        max_grad_norm = 1.0,
        loss_scale = 0,
        no_cuda = False,
        tokenizer = 'iupac',
        resume_from_checkpoint = False,
        num_workers = 8,
        log_level = logging.INFO,
        num_log_iter = 20
    )
    #run_train_distributed(train_args)
    run_train(train_args)


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-o',
                        '--result_dir',
                        required=True,
                        type=str,
                        help='Results directory')
    return parser.parse_args()


if __name__ == '__main__':
    main()
