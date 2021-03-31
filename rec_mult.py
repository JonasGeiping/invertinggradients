"""Run reconstruction in a terminal prompt.
Optional arguments can be found in inversefed/options.py

This CLI can recover the baseline experiments.
"""

import torch
import torchvision

import numpy as np

import inversefed
torch.backends.cudnn.benchmark = inversefed.consts.BENCHMARK

from collections import defaultdict
import datetime
import time
import os
import json
import hashlib
import csv


# Parse input arguments
parser = inversefed.options()
parser.add_argument('--unsigned', action='store_true', help='Use signed gradient descent')
parser.add_argument('--soft_labels', action='store_true', help='Do not use the provided label when using L-BFGS (This can stabilize it).')
parser.add_argument('--lr', default=None, type=float, help='Optionally overwrite default step sizes.')
parser.add_argument('--num_exp', default=10, type=int, help='Number of consecutive experiments')
parser.add_argument('--max_iterations', default=4800, type=int, help='Maximum number of iterations for reconstruction.')
parser.add_argument('--batch_size', default=0, type=int, help='Number of mini batch for federated averaging')
parser.add_argument('--local_lr', default=1e-4, type=float, help='Local learning rate for federated averaging')
args = parser.parse_args()
if args.target_id is None:
    args.target_id = 0
args.save_image = True
args.signed = not args.unsigned


# Parse training strategy
defs = inversefed.training_strategy('conservative')
defs.epochs = args.epochs
# 100% reproducibility?
if args.deterministic:
    image2graph2vec.utils.set_deterministic()


if __name__ == "__main__":
    # Choose GPU device and print status information:
    setup = inversefed.utils.system_startup(args)
    start_time = time.time()

    # Prepare for training

    # Get data:
    loss_fn, trainloader, validloader = inversefed.construct_dataloaders(args.dataset, defs)

    model, model_seed = inversefed.construct_model(args.model, num_classes=10, num_channels=3)
    dm = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_mean'), **setup)[:, None, None]
    ds = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_std'), **setup)[:, None, None]
    model.to(**setup)
    model.eval()

    # Load a trained model?
    if args.trained_model:
        file = f'{args.model}_{args.epochs}.pth'
        try:
            model.load_state_dict(torch.load(os.path.join(args.model_path, file), map_location=setup['device']))
            print(f'Model loaded from file {file}.')
        except FileNotFoundError:
            print('Training the model ...')
            print(repr(defs))
            inversefed.train(model, loss_fn, trainloader, validloader, defs, setup=setup)
            torch.save(model.state_dict(), os.path.join(args.model_path, file))

    # Sanity check: Validate model accuracy
    training_stats = defaultdict(list)
    inversefed.training.training_routine.validate(model, loss_fn, validloader, defs, setup, training_stats)
    name, format = loss_fn.metric()
    print(f'Val loss is {training_stats["valid_losses"][-1]:6.4f}, Val {name}: {training_stats["valid_" + name][-1]:{format}}.')

    if args.optim == 'ours':
        config = dict(signed=args.signed,
                      boxed=True,
                      cost_fn=args.cost_fn,
                      indices=args.indices,
                      weights=args.weights,
                      lr=args.lr if args.lr is not None else 0.1,
                      optim='adam',
                      restarts=args.restarts,
                      max_iterations=args.max_iterations,
                      total_variation=args.tv,
                      init=args.init,
                      filter='none',
                      lr_decay=True,
                      scoring_choice=args.scoring_choice)
    elif args.optim == 'zhu':
        config = dict(signed=False,
                      boxed=False,
                      cost_fn='l2',
                      indices='def',
                      weights='equal',
                      lr=args.lr if args.lr is not None else 1.0,
                      optim='LBFGS',
                      restarts=args.restarts,
                      max_iterations=500,
                      total_variation=args.tv,
                      init=args.init,
                      filter='none',
                      lr_decay=False,
                      scoring_choice=args.scoring_choice)

    # psnr list
    psnrs = []

    # hash configuration

    config_comp = config.copy()
    config_comp['dataset'] = args.dataset
    config_comp['model'] = args.model
    config_comp['trained'] = args.trained_model
    config_comp['num_exp'] = args.num_exp
    config_comp['num_images'] = args.num_images
    config_comp['accumulation'] = args.accumulation
    config_comp['batch_size'] = args.batch_size
    config_comp['local_lr'] = args.trained_model
    config_comp['soft_labels'] = args.soft_labels
    config_hash = hashlib.md5(json.dumps(config_comp, sort_keys=True).encode()).hexdigest()

    print(config_comp)

    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{config_hash}', exist_ok=True)


    target_id = args.target_id
    for i in range(args.num_exp):
        target_id = args.target_id + i * args.num_images
        if args.num_images == 1:
            ground_truth, labels = validloader.dataset[target_id]
            if args.label_flip:
                labels = torch.randint((10,))
            ground_truth, labels = ground_truth.unsqueeze(0).to(**setup), torch.as_tensor((labels,), device=setup['device'])
            target_id_ = target_id + 1
        else:
            ground_truth, labels = [], []
            target_id_ = target_id
            while len(labels) < args.num_images:
                img, label = validloader.dataset[target_id_]
                target_id_ += 1
                if label not in labels:
                    labels.append(torch.as_tensor((label,), device=setup['device']))
                    ground_truth.append(img.to(**setup))

            ground_truth = torch.stack(ground_truth)
            labels = torch.cat(labels)
            if args.label_flip:
                labels = torch.permute(labels)
        img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

        # Run reconstruction
        if args.accumulation == 0:
            target_loss, _, _ = loss_fn(model(ground_truth), labels)
            input_gradient = torch.autograd.grad(target_loss, model.parameters())
            input_gradient = [grad.detach() for grad in input_gradient]

            # Run reconstruction in different precision?
            if args.dtype != 'float':
                if args.dtype in ['double', 'float64']:
                    setup['dtype'] = torch.double
                elif args.dtype in ['half', 'float16']:
                    setup['dtype'] = torch.half
                else:
                    raise ValueError(f'Unknown data type argument {args.dtype}.')
                print(f'Model and input parameter moved to {args.dtype}-precision.')
                dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
                ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
                ground_truth = ground_truth.to(**setup)
                input_gradient = [g.to(**setup) for g in input_gradient]
                model.to(**setup)
                model.eval()

            rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=args.num_images)

            if args.optim == 'zhu' and args.soft_labels:
                rec_machine.iDLG = False
                output, stats = rec_machine.reconstruct(input_gradient, None, img_shape=img_shape, dryrun=args.dryrun)
            else:
                output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=img_shape, dryrun=args.dryrun)

        else:
            local_gradient_steps = args.accumulation
            local_lr = args.local_lr
            batch_size = args.batch_size
            input_parameters = inversefed.reconstruction_algorithms.loss_steps(model, ground_truth,
                                                                               labels,
                                                                               lr=local_lr,
                                                                               local_steps=local_gradient_steps, use_updates=True, batch_size=batch_size)
            input_parameters = [p.detach() for p in input_parameters]

            # Run reconstruction in different precision?
            if args.dtype != 'float':
                if args.dtype in ['double', 'float64']:
                    setup['dtype'] = torch.double
                elif args.dtype in ['half', 'float16']:
                    setup['dtype'] = torch.half
                else:
                    raise ValueError(f'Unknown data type argument {args.dtype}.')
                print(f'Model and input parameter moved to {args.dtype}-precision.')
                ground_truth = ground_truth.to(**setup)
                dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
                ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
                input_parameters = [g.to(**setup) for g in input_parameters]
                model.to(**setup)
                model.eval()

            rec_machine = inversefed.FedAvgReconstructor(model, (dm, ds), local_gradient_steps,
                                                         local_lr, config,
                                                         num_images=args.num_images, use_updates=True,
                                                         batch_size=batch_size)
            output, stats = rec_machine.reconstruct(input_parameters, labels, img_shape=img_shape, dryrun=args.dryrun)



        # Compute stats and save to a table:
        output_den = torch.clamp(output * ds + dm, 0, 1)
        ground_truth_den = torch.clamp(ground_truth * ds + dm, 0, 1)
        feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
        test_mse = (output_den - ground_truth_den).pow(2).mean().item()
        test_psnr = inversefed.metrics.psnr(output_den, ground_truth_den, factor=1)
        print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |")

        inversefed.utils.save_to_table(f'results/{config_hash}', name=f'mul_exp_{args.name}', dryrun=args.dryrun,

                                       config_hash=config_hash,
                                       model=args.model,
                                       dataset=args.dataset,
                                       trained=args.trained_model,
                                       accumulation=args.accumulation,
                                       restarts=args.restarts,
                                       OPTIM=args.optim,
                                       cost_fn=args.cost_fn,
                                       indices=args.indices,
                                       weights=args.weights,
                                       scoring=args.scoring_choice,
                                       init=args.init,
                                       tv=args.tv,

                                       rec_loss=stats["opt"],
                                       psnr=test_psnr,
                                       test_mse=test_mse,
                                       feat_mse=feat_mse,

                                       target_id=target_id,
                                       seed=model_seed,
                                       dtype=setup['dtype'],
                                       epochs=defs.epochs,
                                       val_acc=training_stats["valid_" + name][-1],
                                       )


        # Save the resulting image
        if args.save_image and not args.dryrun:
            output_denormalized = torch.clamp(output * ds + dm, 0, 1)
            for j in range(args.num_images):
                filename = (f'{i*args.num_images+j}.png')

                torchvision.utils.save_image(output_denormalized[j:j + 1, ...],
                                             os.path.join(f'results/{config_hash}', filename))

        # Save psnr values
        psnrs.append(test_psnr)
        inversefed.utils.save_to_table(f'results/{config_hash}', name='psnrs', dryrun=args.dryrun, target_id=target_id, psnr=test_psnr)

        # Update target id
        target_id = target_id_


    # psnr statistics
    psnrs = np.nan_to_num(np.array(psnrs))
    psnr_mean = psnrs.mean()
    psnr_std = np.std(psnrs)
    psnr_max = psnrs.max()
    psnr_min = psnrs.min()
    psnr_median = np.median(psnrs)
    timing = datetime.timedelta(seconds=time.time() - start_time)
    inversefed.utils.save_to_table(f'results/{config_hash}', name='psnr_stats', dryrun=args.dryrun,
                                   number_of_samples=len(psnrs),
                                   timing=str(timing),
                                   mean=psnr_mean,
                                   std=psnr_std,
                                   max=psnr_max,
                                   min=psnr_min,
                                   median=psnr_median)

    config_exists = False
    if os.path.isfile('results/table_configs.csv'):
        with open('results/table_configs.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                if row[-1] == config_hash:
                    config_exists = True
                    break

    if not config_exists:
        inversefed.utils.save_to_table('results', name='configs', dryrun=args.dryrun,
                                       config_hash=config_hash,
                                       **config_comp,
                                       number_of_samples=len(psnrs),
                                       timing=str(timing),
                                       mean=psnr_mean,
                                       std=psnr_std,
                                       max=psnr_max,
                                       min=psnr_min,
                                       median=psnr_median)

    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}')
    print('-------------Job finished.-------------------------')
