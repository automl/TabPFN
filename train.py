import os
import itertools
import argparse
import time
import datetime
import yaml
from contextlib import nullcontext


import torch
from torch import nn

import utils
from transformer import TransformerModel
from utils import get_cosine_schedule_with_warmup, get_openai_lr, StoreDictKeyPair, get_weighted_single_eval_pos_sampler, get_uniform_single_eval_pos_sampler
import priors
import encoders
import positional_encodings
from utils import init_dist
from torch.cuda.amp import autocast

class Losses():
    gaussian = nn.GaussianNLLLoss(full=True, reduction='none')
    mse = nn.MSELoss(reduction='none')
    ce = lambda weight : nn.CrossEntropyLoss(reduction='none', weight=weight)
    bce = nn.BCEWithLogitsLoss(reduction='none')


def train(priordataloader_class, criterion, encoder_generator, emsize=200, nhid=200, nlayers=6, nhead=2, dropout=0.2,
          epochs=10, steps_per_epoch=100, batch_size=200, bptt=10, lr=None, weight_decay=0.0, warmup_epochs=10, input_normalization=False,
          y_encoder_generator=None, pos_encoder_generator=None, decoder=None, extra_prior_kwargs_dict={}, scheduler=get_cosine_schedule_with_warmup,
          load_weights_from_this_state_dict=None, validation_period=10, single_eval_pos_gen=None, bptt_extra_samples=None, gpu_device='cuda:0',
          aggregate_k_gradients=1, verbose=True, style_encoder_generator=None, check_is_compatible=True, epoch_callback=None,
          initializer=None, initialize_with_model=None, train_mixed_precision=False, total_available_time_in_s=None, normalize_labels=True, **model_extra_args
          ):
    assert (epochs is None) != (total_available_time_in_s is None)
    start_of_training = time.time()
    device = gpu_device if torch.cuda.is_available() else 'cpu:0'
    print(f'Using {device} device')
    using_dist, rank, device = init_dist(device)
    bptt_sampler = (lambda : single_eval_pos_gen() + bptt_extra_samples if callable(single_eval_pos_gen) else single_eval_pos_gen + bptt_extra_samples) if bptt_extra_samples is not None else bptt
    dl = priordataloader_class(num_steps=steps_per_epoch, batch_size=batch_size, seq_len=bptt_sampler, seq_len_maximum=bptt+(bptt_extra_samples if bptt_extra_samples else 0), device=device, **extra_prior_kwargs_dict)
    if dl.fuse_x_y:
        raise Exception("Illegal parameter")

    encoder = encoder_generator(dl.num_features+1 if dl.fuse_x_y else dl.num_features,emsize)
    style_def = next(iter(dl))[0][0] # This is (style, x, y), target with x and y with batch size

    style_encoder = style_encoder_generator(hyperparameter_definitions=style_def[0], em_size=emsize) if (style_def is not None) else None
    n_out = dl.num_outputs
    if isinstance(criterion, nn.GaussianNLLLoss):
        n_out *= 2
    elif isinstance(criterion, nn.CrossEntropyLoss):
        n_out *= criterion.weight.shape[0]
    model = TransformerModel(encoder, n_out, emsize, nhead, nhid, nlayers, dropout, style_encoder=style_encoder,
                             y_encoder=y_encoder_generator(dl.num_outputs, emsize), input_normalization=input_normalization,
                             pos_encoder=(pos_encoder_generator or positional_encodings.NoPositionalEncoding)(emsize, bptt*2),
                             decoder=decoder, init_method=initializer, **model_extra_args
                             )
    model.criterion = criterion
    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict)
    if initialize_with_model is not None:
        model.init_from_small_model(initialize_with_model)

    print(f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters")

    try:
        for (k, v), (k2, v2) in zip(model.state_dict().items(), initialize_with_model.state_dict().items()):
            print(k, ((v - v2) / v).abs().mean(), v.shape)
    except Exception:
        pass

    model.to(device)
    if using_dist:
        print("Distributed training")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)


    # learning rate
    if lr is None:
        lr = get_openai_lr(model)
        print(f"Using OpenAI max lr of {lr}.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler(optimizer, warmup_epochs, epochs if epochs is not None else 100) # when training for fixed time lr schedule takes 100 steps

    def train_step():
        model.train()  # Turn on the train mode
        total_loss = 0.
        total_positional_losses = 0.
        total_positional_losses_recorded = 0
        before_get_batch = time.time()
        assert len(dl) % aggregate_k_gradients == 0, 'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'
        valid_batch_steps = 0.0
        for batch, (data, targets) in enumerate(dl):
            if using_dist and not (batch % aggregate_k_gradients == aggregate_k_gradients - 1):
                cm = model.no_sync()
                #print(f'p={rank}, no_sync', force=True)
            else:
                cm = nullcontext()
                #print(f'p={rank}, sync', force=True)
            with cm:
                time_to_get_batch = time.time() - before_get_batch
                before_forward = time.time()
                if bptt_extra_samples is None:
                    single_eval_pos = single_eval_pos_gen() if callable(single_eval_pos_gen) else single_eval_pos_gen
                else:
                    single_eval_pos = targets.shape[0] - bptt_extra_samples

                is_compatible = torch.ones((targets.shape[1])).bool()
                if check_is_compatible or normalize_labels:
                    for b in range(targets.shape[1]):
                        targets_in_train = torch.unique(targets[:single_eval_pos, b], sorted=True)
                        targets_in_eval = torch.unique(targets[single_eval_pos:, b], sorted=True)

                        if check_is_compatible:
                            is_compatible[b] = len(targets_in_train) == len(targets_in_eval) and (targets_in_train == targets_in_eval).all()
                            is_compatible[b] = is_compatible[b] and len(targets_in_train) > 1

                        # Set targets to range starting from 0 (e.g. targets 0, 2, 5, 2 will be converted to 0, 1, 2, 1)
                        if normalize_labels:
                            targets[:, b] = (targets[:, b] > torch.unique(targets[:, b]).unsqueeze(1)).sum(axis=0).unsqueeze(0)
                valid_batch_steps += is_compatible.float().mean()
                is_compatible = is_compatible.to(device)
                #if using_dist and check_is_compatible:
                #    print('step share before reduce',curr_step_share, force=True)
                #    curr_step_share = curr_step_share.to(device)
                #    torch.distributed.all_reduce_multigpu([curr_step_share], op=torch.distributed.ReduceOp.SUM)
                #    curr_step_share = curr_step_share.cpu() / torch.distributed.get_world_size()
                #    print('step share after reduce',curr_step_share, torch.distributed.get_world_size(), force=True)

                # If style is set to None, it should not be transferred to device
                output = model(tuple(e.to(device) if torch.is_tensor(e) else e for e in data) if isinstance(data, tuple) else data.to(device)
                               , single_eval_pos=single_eval_pos)

                forward_time = time.time() - before_forward

                #output, targets = output[:, is_compatible], targets[:, is_compatible]

                if single_eval_pos is not None:
                    targets = targets[single_eval_pos:]
                if isinstance(criterion, nn.GaussianNLLLoss):
                    assert output.shape[-1] == 2, \
                        'need to write a little bit of code to handle multiple regression targets at once'

                    mean_pred = output[..., 0]
                    var_pred = output[..., 1].abs()
                    losses = criterion(mean_pred.flatten(), targets.to(device).flatten(), var=var_pred.flatten())
                elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                    losses = criterion(output.flatten(), targets.to(device).flatten())
                elif isinstance(criterion, (nn.CrossEntropyLoss)):
                    #print(n_out, targets.min(), targets.max(), force=True)
                    losses = criterion(output.reshape(-1, n_out), targets.to(device).long().flatten())
                else:
                    losses = criterion(output.reshape(-1, n_out), targets.to(device).flatten())
                losses = losses.view(*output.shape[0:2])
                loss = losses.mean(0) @ is_compatible.float() / losses.shape[1]
                #loss = torch_nanmean(losses, axis=[0, 1]) * is_compatible.float().mean()
                # not sure whether we can go without the nan checks.

                loss.backward()

                if ((batch % aggregate_k_gradients == aggregate_k_gradients - 1) and (not check_is_compatible or using_dist))\
                        or (valid_batch_steps >= aggregate_k_gradients and (check_is_compatible and not using_dist)):
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.div_(valid_batch_steps)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    try:
                        optimizer.step()
                    except:
                        print("Invalid optimization step encountered")
                    optimizer.zero_grad()
                    valid_batch_steps = 0.0

                step_time = time.time() - before_forward

                if not torch.isnan(loss):
                    total_loss += loss.item()
                    total_positional_losses += losses.mean(1).cpu().detach() if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)*loss.cpu().detach()

                    total_positional_losses_recorded += torch.ones(bptt) if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)

            before_get_batch = time.time()
        return total_loss / steps_per_epoch, (
                    total_positional_losses / total_positional_losses_recorded).tolist(), time_to_get_batch, forward_time, step_time

    best_val_loss = float("inf")
    best_model = None
    total_loss = float('inf')
    total_positional_losses = float('inf')
    try:
        for epoch in (range(1, epochs + 1) if epochs is not None else itertools.count(1)):

            epoch_start_time = time.time()
            if train_mixed_precision:
                with autocast():
                    total_loss, total_positional_losses, time_to_get_batch, forward_time, step_time = train_step()
            else:
                total_loss, total_positional_losses, time_to_get_batch, forward_time, step_time = train_step()
            if hasattr(dl, 'validate') and epoch % validation_period == 0:
                with torch.no_grad():
                    val_score = dl.validate(model)
            else:
                val_score = None

            if verbose:
                print('-' * 89)
                print(
                    f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {total_loss:5.2f} | '
                    f"pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])}, lr {scheduler.get_last_lr()[0]}"
                    f' data time {time_to_get_batch:5.2f} step time {step_time:5.2f}'
                    f' forward time {forward_time:5.2f}' + (f'val score {val_score}' if val_score is not None else ''))
                print('-' * 89)

            # stepping with wallclock time based scheduler
            current_time = time.time()
            if epoch_callback is not None and rank == 0:
                epoch_callback(model, epoch / epochs if total_available_time_in_s is None else  # noqa
                (current_time - start_of_training) / total_available_time_in_s  # noqa
                               )
            if epochs is None and (current_time - start_of_training) > total_available_time_in_s:  # noqa
                break
            if epochs is None:
                scheduler.step((current_time - epoch_start_time) / total_available_time_in_s * 100)
            else:
                scheduler.step()
    except KeyboardInterrupt:
        pass

    return total_loss, total_positional_losses, model.to('cpu'), dl

def _parse_args(config_parser, parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


if __name__ == '__main__':
    config_parser = argparse.ArgumentParser(description='Only used as a first parser for the config file path.')
    config_parser.add_argument('--config')
    parser = argparse.ArgumentParser()
    parser.add_argument('prior')
    parser.add_argument('--loss_function', default='barnll')
    # Optional Arg's for `--loss_function barnll`
    parser.add_argument('--min_y', type=float, help='barnll can only model y in strict ranges, this is the minimum y can take.')
    parser.add_argument('--max_y', type=float, help='barnll can only model y in strict ranges, this is the maximum y can take.')
    parser.add_argument('--num_buckets', default=100, type=int)
    #parser.add_argument('--num_features', default=None, type=int, help='Specify depending on the prior.')
    parser.add_argument("--extra_prior_kwargs_dict", default={'fuse_x_y': False}, dest="extra_prior_kwargs_dict", action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL", help='Specify depending on the prior.')
    parser.add_argument('--encoder', default='linear', type=str, help='Specify depending on the prior.')
    parser.add_argument('--y_encoder', default='linear', type=str, help='Specify depending on the prior. You should specify this if you do not fuse x and y.')
    parser.add_argument('--pos_encoder', default='sinus', type=str, help='Specify depending on the prior.')
    parser.add_argument('--bptt', default=10, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', default=50, type=int)
    parser.add_argument('--validation_period', default=10, type=int)
    parser.add_argument('--permutation_invariant_max_eval_pos', default=None, type=int, help='Set this to an int to ')
    parser.add_argument('--permutation_invariant_sampling', default='weighted', help="Only relevant if --permutation_invariant_max_eval_pos is set.")

    # these can likely be mostly left at defaults
    parser.add_argument('--emsize', default=512, type=int) # sometimes even larger is better e.g. 1024
    parser.add_argument('--nlayers', default=6, type=int)
    parser.add_argument('--nhid', default=None, type=int) # 2*emsize is the default
    parser.add_argument('--nhead', default=4, type=int) # nhead = emsize / 64 in the original paper
    parser.add_argument('--dropout', default=.0, type=float)
    parser.add_argument('--steps_per_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--lr', '--learning_rate', default=.001, type=float) # try also .0003, .0001, go lower with lower batch size

    args, _ = _parse_args(config_parser, parser)

    if args.nhid is None:
        args.nhid = 2*args.emsize

    prior = args.__dict__.pop('prior')

    if prior == 'gp':
        prior = priors.fast_gp.DataLoader
    elif prior == 'ridge':
        prior = priors.ridge.DataLoader
    elif prior == 'stroke':
        prior = priors.stroke.DataLoader
    elif prior == 'mix_gp':
        prior = priors.fast_gp_mix.DataLoader
    else:
        raise NotImplementedError(f'Prior == {prior}.')

    loss_function = args.__dict__.pop('loss_function')

    criterion = nn.GaussianNLLLoss(reduction='none', full=True)
    classificiation_criterion = nn.CrossEntropyLoss(reduction='none')
    num_buckets = args.__dict__.pop('num_buckets')
    max_y = args.__dict__.pop('max_y')
    min_y = args.__dict__.pop('min_y')
    # criterion = nn.MSELoss(reduction='none')

    def get_y_sample():
        dl = prior(num_steps=1, batch_size=args.batch_size * args.steps_per_epoch, seq_len=args.bptt, device=device,
                   **args.extra_prior_kwargs_dict)
        y_sample = next(iter(dl))[-1]
        print(f'Creating Bar distribution with borders from y sample of size {y_sample.numel()}')
        return y_sample

    if loss_function == 'ce':
        criterion = nn.CrossEntropyLoss(reduction='none')
    elif loss_function == 'gaussnll':
        criterion = nn.GaussianNLLLoss(reduction='none', full=True)
    elif loss_function == 'mse':
        criterion = nn.MSELoss(reduction='none')
    elif loss_function == 'barnll':
        criterion = BarDistribution(borders=get_bucket_limits(num_buckets, full_range=(min_y,max_y)))
    elif loss_function == 'adaptivebarnll':
        borders = get_bucket_limits(num_buckets, ys=get_y_sample(), full_range=(min_y,max_y))
        criterion = BarDistribution(borders=borders)
    elif loss_function == 'adaptivefullsupportbarnll':
        assert min_y is None and max_y is None, "Please do not specify `min_y` and `max_y` with `unboundedadaptivebarnll`."
        borders = get_bucket_limits(num_buckets, ys=get_y_sample())
        criterion = FullSupportBarDistribution(borders=borders)
    else:
        raise NotImplementedError(f'loss_function == {loss_function}.')



    encoder = args.__dict__.pop('encoder')
    y_encoder = args.__dict__.pop('y_encoder')

    def get_encoder_generator(encoder):
        if encoder == 'linear':
            encoder_generator = encoders.Linear
        elif encoder == 'mlp':
            encoder_generator = encoders.MLP
        elif encoder == 'positional':
            encoder_generator = encoders.Positional
        else:
            raise NotImplementedError(f'A {encoder} encoder is not valid.')
        return encoder_generator

    encoder_generator = get_encoder_generator(encoder)
    y_encoder_generator = get_encoder_generator(y_encoder)

    pos_encoder = args.__dict__.pop('pos_encoder')

    if pos_encoder == 'none':
        pos_encoder_generator = None
    elif pos_encoder == 'sinus':
        pos_encoder_generator = positional_encodings.PositionalEncoding
    elif pos_encoder == 'learned':
        pos_encoder_generator = positional_encodings.LearnedPositionalEncoding
    elif pos_encoder == 'paired_scrambled_learned':
        pos_encoder_generator = positional_encodings.PairedScrambledPositionalEncodings
    else:
        raise NotImplementedError(f'pos_encoer == {pos_encoder} is not valid.')

    permutation_invariant_max_eval_pos = args.__dict__.pop('permutation_invariant_max_eval_pos')
    permutation_invariant_sampling = args.__dict__.pop('permutation_invariant_sampling')
    if permutation_invariant_max_eval_pos is not None:
        if permutation_invariant_sampling == 'weighted':
            get_sampler = get_weighted_single_eval_pos_sampler
        elif permutation_invariant_sampling == 'uniform':
            get_sampler = get_uniform_single_eval_pos_sampler
        else:
            raise ValueError()
        args.__dict__['single_eval_pos_gen'] = get_sampler(permutation_invariant_max_eval_pos)


    print("ARGS for `train`:", args.__dict__)

    train(prior, criterion, encoder_generator,
          y_encoder_generator=y_encoder_generator, pos_encoder_generator=pos_encoder_generator,
          **args.__dict__)

