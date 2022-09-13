"""
Ray-TAPE integration
"""
import json
import logging
from pathlib import Path
import typing

from tape import utils, visualization, errors
from tape.registry import registry
from tape.training import BackwardRunner, run_train_epoch, run_valid_epoch

from ray import tune


def run_train(model_type: str,
              task: str,
              learning_rate: float = 1e-4,
              batch_size: int = 1024,
              num_train_epochs: int = 10,
              num_log_iter: int = 20,
              fp16: bool = False,
              warmup_steps: int = 10000,
              gradient_accumulation_steps: int = 1,
              loss_scale: int = 0,
              max_grad_norm: float = 1.0,
              exp_name: typing.Optional[str] = None,
              from_pretrained: typing.Optional[str] = None,
              log_dir: str = './logs',
              eval_freq: int = 1,
              save_freq: typing.Union[int, str] = 'improvement',
              model_config_file: typing.Optional[str] = None,
              data_dir: str = './data',
              output_dir: str = './results',
              no_cuda: bool = False,
              seed: int = 42,
              local_rank: int = -1,
              tokenizer: str = 'iupac',
              num_workers: int = 8,
              debug: bool = False,
              log_level: typing.Union[str, int] = logging.INFO,
              patience: int = -1,
              resume_from_checkpoint: bool = False) -> float:
    """
    Synchronize paths and set up communication with Ray.
    """
    # SETUP AND LOGGING CODE #
    input_args = locals()
    device, n_gpu, is_master = utils.setup_distributed(
        local_rank, no_cuda)

    exp_dir = utils.get_expname(exp_name, task, model_type)

    utils.barrier_if_distributed()
    # utils.setup_logging(local_rank, save_path, log_level)
    utils.set_random_seeds(seed, n_gpu)

    train_dataset = utils.setup_dataset(task, data_dir, 'train', tokenizer)
    valid_dataset = utils.setup_dataset(task, data_dir, 'valid', tokenizer)
    train_loader = utils.setup_loader(
        train_dataset, batch_size, local_rank, n_gpu,
        gradient_accumulation_steps, num_workers)
    valid_loader = utils.setup_loader(
        valid_dataset, batch_size, local_rank, n_gpu,
        gradient_accumulation_steps, num_workers)

    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        train_dataset, batch_size, num_train_epochs)

    model = registry.get_task_model(model_type, task, model_config_file, from_pretrained)
    model = model.to(device)
    optimizer = utils.setup_optimizer(model, learning_rate)
    with tune.checkpoint_dir(step=0) as checkpoint_dir:
        viz = visualization.get(str(Path(checkpoint_dir).parent), exp_dir,
                                local_rank, debug=debug)
    viz.log_config(input_args)
    viz.log_config(model.config.to_dict())
    viz.watch(model)

    print(f"device: {device} "
          f"n_gpu: {n_gpu}, "
          f"distributed_training: {local_rank != -1}, "
          f"16-bits training: {fp16}")

    runner = BackwardRunner(
        model, optimizer, gradient_accumulation_steps, device, n_gpu,
        fp16, local_rank, max_grad_norm, warmup_steps, num_train_optimization_steps)

    runner.initialize_fp16()
    if resume_from_checkpoint:
        checkpoint = str(Path(from_pretrained))
        start_epoch = runner.resume_from_checkpoint(checkpoint)
    else:
        start_epoch = 0
    runner.initialize_distributed_model()

    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        train_dataset, batch_size, num_train_epochs)
    is_master = local_rank in (-1, 0)

    if isinstance(save_freq, str) and save_freq != 'improvement':
        raise ValueError(
            f"Only recongized string value for save_freq is 'improvement'"
            f", received: {save_freq}")

    if save_freq == 'improvement' and eval_freq <= 0:
        raise ValueError("Cannot set save_freq to 'improvement' and eval_freq < 0")

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Batch size = %d", batch_size)
    print("  Num epochs = %d", num_train_epochs)
    print("  Num train steps = %d", num_train_optimization_steps)
    print("  Num parameters = %d", num_trainable_parameters)

    best_val_loss = float('inf')
    num_evals_no_improvement = 0

    def do_save(epoch_id: int, num_evals_no_improvement: int) -> bool:
        if not is_master:
            return False
        if isinstance(save_freq, int):
            return ((epoch_id + 1) % save_freq == 0) or ((epoch_id + 1) == num_train_epochs)
        else:
            return num_evals_no_improvement == 0

    utils.barrier_if_distributed()

    # ACTUAL TRAIN/EVAL LOOP #
    with utils.wrap_cuda_oom_error(local_rank, batch_size, n_gpu, gradient_accumulation_steps):
        for epoch_id in range(start_epoch, num_train_epochs):
            run_train_epoch(epoch_id, train_loader, runner,
                            viz, num_log_iter, gradient_accumulation_steps)
            if eval_freq > 0 and (epoch_id + 1) % eval_freq == 0:
                val_loss, _ = run_valid_epoch(epoch_id, valid_loader, runner, viz, is_master)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    num_evals_no_improvement = 0
                else:
                    num_evals_no_improvement += 1

            # Save trained model
            if do_save(epoch_id, num_evals_no_improvement):
                print("** ** * Saving trained model ** ** * ")
                # Only save the model itself

                with tune.checkpoint_dir(step=epoch_id) as checkpoint_dir:
                    checkpoint_path = Path(checkpoint_dir)
                    if is_master:
                        # save all the hidden parameters.
                        checkpoint_path.mkdir(parents=True, exist_ok=True)
                        with (checkpoint_path / 'args.json').open('w') as f:
                            json.dump(input_args, f)
                    checkpoint_path = str(checkpoint_path)
                    runner.save_state(checkpoint_path, epoch_id)

                print(f"Saving model checkpoint to {checkpoint_path}")

            utils.barrier_if_distributed()
            if patience > 0 and num_evals_no_improvement >= patience:
                print(f"Finished training at epoch {epoch_id} because no "
                      f"improvement for {num_evals_no_improvement} epochs.")
                print(f"Best Val Loss: {best_val_loss}")
                if local_rank != -1:
                    # If you're distributed, raise this error. It sends a signal to
                    # the master process which lets it kill other processes and terminate
                    # without actually reporting an error. See utils/distributed_utils.py
                    # for the signal handling code.
                    raise errors.EarlyStopping
                else:
                    break
    print(f"Finished training after {num_train_epochs} epochs.")
    if best_val_loss != float('inf'):
        print(f"Best Val Loss: {best_val_loss}")

    return best_val_loss
