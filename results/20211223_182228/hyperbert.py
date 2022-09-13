# Hyperparameter search using Ray
# This script is essentially just experiment-specific configuration
import argparse
import inspect
import json
import tempfile
from pathlib import Path
import sys

from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest import ConcurrencyLimiter

from scripts.general.util import LogCapture
from scripts.model import ray_tape

_this_filename = inspect.getframeinfo(inspect.currentframe()).filename
_this_path = Path(_this_filename).parent.resolve()


def main():
    args = get_args()

    search_space = {
        'learning_rate': tune.loguniform(1e-8, 1e-6),
        'num_train_epochs': tune.choice([500, 800]),
        'batch_size': tune.choice([16, 32, 64]),
        'num_hidden_layers': tune.choice([8, 12, 16, 20, 24]),
        'num_attention_heads': tune.choice([4, 8, 16, 20, 24]),     # LCM = 240
        'hidden_size': tune.choice([960, 1200, 1680, 2400, 3120]),  # multiple of attention heads
        'intermediate_size': tune.choice([2048, 3072, 5120]),
        'hidden_dropout_prob': tune.choice([0.0]),
        'attention_probs_dropout_prob': tune.choice([0.0]),
        'hidden_act': tune.choice(['relu']),
        'resume': args.resume
    }

    algo_bohb = ConcurrencyLimiter(TuneBOHB(metric="mean_loss", mode="min"),
                                   max_concurrent=4)
    scheduler_bohb = HyperBandForBOHB(
        time_attr="training_iteration",
        metric="mean_loss",
        mode="min",
        max_t=100)

    analysis = tune.run(
        training_wrapper,
        scheduler=scheduler_bohb,
        search_alg=algo_bohb,
        num_samples=40,
        keep_checkpoints_num=2,
        checkpoint_score_attr='min-mean_loss',
        resources_per_trial={"gpu": 2, 'cpu': 2},
        config=search_space,
        name='hyperbert',
        resume=args.resume,
        max_failures=1,
        local_dir=str(_this_path / 'ray_results')
    )

    print("Best config: ",
          analysis.get_best_config(metric="mean_loss", mode="min"))
    analysis.results_df.to_csv(_this_path / 'ray_tune_results.csv', index=False)


def training_wrapper(hp: dict, checkpoint_dir=None):
    """
    Building on top of TAPEs code flow.
    The TAPE Bert reads hyperparams from a file: writing `hp` to a temp file.
    """
    # This registers the learning task (including data) for TAPE
    from scripts.tape_elements import SequenceAbundanceDataset, tape_model_name

    arch = {
        "vocab_size": 30,
        "hidden_size": hp['hidden_size'],
        "num_hidden_layers": hp['num_hidden_layers'],
        "num_attention_heads": hp['num_attention_heads'],
        "intermediate_size": hp['intermediate_size'],
        "hidden_act": hp['hidden_act'],
        "hidden_dropout_prob": hp['hidden_dropout_prob'],
        "attention_probs_dropout_prob": hp['attention_probs_dropout_prob'],
        "max_position_embeddings": 1024,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12
    }

    with tempfile.NamedTemporaryFile('w') as arch_file:
        json.dump(arch, arch_file)
        arch_file.flush()
        arch_filename = arch_file.name

        try:
            best_score = ray_tape.run_train(
                model_type=tape_model_name,
                task='learn_abundance',

                learning_rate=hp['learning_rate'],
                num_train_epochs=hp['num_train_epochs'],
                batch_size=hp['batch_size'],
                gradient_accumulation_steps=hp['batch_size'] // 2,
                patience=int(0.1 * hp['num_train_epochs']),

                data_dir=str(_this_path / 'data'),
                log_dir=str(_this_path / 'hyperlogs'),
                model_config_file=arch_filename,
                save_freq='improvement',

                resume_from_checkpoint=hp['resume'],
                from_pretrained=checkpoint_dir
            )
            tune.report(mean_loss=best_score)

        except Exception as e:
            warning_msg = '\n'.join([f'Error in TAPE training:',
                                     str(e.__class__.__name__),
                                     str(e),
                                     str(sys.exc_info()[0])])
            print(warning_msg)


def get_best_score_for_training(tape_logger: LogCapture) -> float:
    for msg in reversed(tape_logger.output):
        if msg[0] == 35:
            return float(msg[1].split(': ')[-1])
    raise Exception('hyperbert.get_best_score_for_training(): '
                    'Could not read best score from TAPE training log')


def get_args():
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument('-r',
                        '--resume',
                        required=False,
                        action='store_true',
                        help='Whether Ray should resume from checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    main()
