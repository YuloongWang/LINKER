import argparse
import os
import random
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid
import torch
import torch.autograd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import csv
from tqdm import tqdm
import wandb

from dataset import Datasets_SZ, Datasets_ASD
from model import Ourmutimodel
from logger import Logger

import warnings
warnings.filterwarnings("ignore")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='../')
    parser.add_argument('--datname', type=str, default='SZ', help='SZ, ASD')  # for datasets
    parser.add_argument('--csv_name', type=str, default='v15_SZ', help='folder name')
    parser.add_argument('--project', type=str, default='Promptlearning', help='wandb project name')

    parser.add_argument('--cuda', type=int, default=3, help='')
    parser.add_argument('--seed', type=int, default=0, help='random seed setting')
    parser.add_argument('--n_class', type=int, default=2)
    parser.add_argument('--n_fold', type=int, default=5)
    parser.add_argument('--n_epoch', type=int, default=30, help='Max number of epochs to train.')
    parser.add_argument('--patience', type=int, default=1000, help='Extra iterations before early-stopping.')

    parser.add_argument('--lr', type=float, default=0.00003)  # 0.001
    parser.add_argument('--max_lr', type=float, default=0.005)  # 0.005
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay.')

    parser.add_argument('--outer', action='store_true', default=False, help="train only outer")
    parser.add_argument('--wandb', action='store_true', default=False, help="enable wandb")
    parser.add_argument('--resume', action='store_true', default=False, help="resume checkpoint")
    parser.add_argument('--savefile', action='store_true', default=True, help='save model')
    parser.add_argument('--savecheckpoint', action='store_true', default=False, help='save checkpoint')


    parser.add_argument('--batch_size', type=int, default=32)  # 32
    parser.add_argument('--dropout', type=float, default=0.5)  # 0.5

    # FCN
    parser.add_argument('--fcn_layer', type=int, default=4)  # 3
    parser.add_argument('--fcn_head', type=int, default=1)
    # MSN
    parser.add_argument('--msn_layer', type=int, default=2)  # 1
    parser.add_argument('--msn_head', type=int, default=4)

    argv = parser.parse_args()
    return argv


def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model(model, args, params):
    return model


def load_optimizer_scheduler(model, args, params, dataloader_train):
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.n_epoch,
    #                                                 steps_per_epoch=len(dataloader_train), pct_start=0.2,
    #                                                 div_factor=args.max_lr / args.lr, final_div_factor=1000)
    return optimizer, scheduler


def loss_fn(recon, criterion, args, params):
    Y_label = recon['Y_label']  # [batch](0,1)
    Y_pred = recon['Y_pred']  # [batch, class]
    loss_ce = criterion(Y_pred, Y_label)
    return loss_ce


def step(model, dataloader=None, criterion=None, optimizer=None, scheduler=None, args=None, params=None, idxs=None):
    if optimizer is not None: model.train()
    else: model.eval()

    running_loss = 0.0
    preds, probs = [], []
    labels = []
    attn_fcn, attn_msn = [], []
    for i, multidata in enumerate(tqdm(dataloader, ncols=80, desc=idxs)):
        if optimizer is not None:
            output = model(multidata['fcn-msn'])
        else:
            with torch.no_grad():
                output = model(multidata['fcn-msn'])
        # print(model.alpha.grad, model.gnn_layers[0].epsilon.grad)
        label = (multidata['label'] % 2).to(args.device)
        output['Y_label'] = label
        loss = loss_fn(output, criterion, args, params)

        logit = output['Y_pred']
        preds.append(logit.argmax(1).detach().cpu().numpy())
        probs.append(logit.softmax(1).detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())

        # optimize model
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            # attn_fcn.append(output['attn_fcn'].detach().cpu().numpy())
            # attn_msn.append(output['attn_msn'].detach().cpu().numpy())
            if isinstance(output['attn_fcn'], dict):
                attn_fcn_dict = {}
                for key, value in output['attn_fcn'].items():
                    attn_fcn_dict[key] = value.detach().cpu().numpy()
                attn_fcn.append(attn_fcn_dict)
            else:
                attn_fcn.append(output['attn_fcn'].detach().cpu().numpy())

            if isinstance(output['attn_msn'], dict):
                attn_msn_dict = {}
                for key, value in output['attn_msn'].items():
                    attn_msn_dict[key] = value.detach().cpu().numpy()
                attn_msn.append(attn_msn_dict)
            else:
                attn_msn.append(output['attn_msn'].detach().cpu().numpy())

        running_loss += loss.detach().cpu().numpy()
    loss_dict = {'loss': running_loss / len(labels)}

    preds = np.concatenate(preds)
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    attn = {'attn_fcn': attn_fcn, 'attn_msn': attn_msn}
    return loss_dict, preds, probs, labels, attn


args = parse()
set_rng_seed(args.seed)
args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
print(f'use {args.device}')
args.data_path = os.path.join(args.dir, f'process/{args.datname}_Process/')

if args.datname == 'SZ': dataset = Datasets_SZ(args)
elif args.datname == 'ASD': dataset = Datasets_ASD(args)
else: assert False

priori = dataset.priori_dict
args.fcn_node = dataset.fcn[0].shape[0]
args.msn_node = dataset.msn[0].shape[0]

# class_counts = torch.bincount(torch.tensor(dataset.labels % 2))
# print(class_counts)
# weight = len(dataset.labels) / class_counts.float() / args.n_class

results_path = os.path.join(args.dir, 'models', args.csv_name)
if not os.path.exists(results_path): os.mkdir(results_path)
with open(os.path.join(results_path, 'argv.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(vars(args).items())

param_grid = {
    'batch_size': [32],  # 32
    'dropout': [0.5],  # 0.5
    # FCN
    'fcn_layer': [1, 2, 3, 4],
    'fcn_head': [1, 2, 4],
    # MSN
    'msn_layer': [1, 2, 3, 4],
    'msn_head': [1, 2, 4],
}

if not args.outer:
    dict_all = {
        'best_inner_params': [],
        'best_inner_val_acc': [],
        'best_inner_val_loss': [],
        'average_best_epoch': [],
    }

# resume checkpoint if file exists
if os.path.isfile(os.path.join(results_path, f'checkpoint.pth')) and args.resume:
    print('resuming checkpoint experiment')
    checkpoint = torch.load(os.path.join(results_path, f'checkpoint.pth'), map_location=args.device)
else:
    checkpoint = {
        'fold_outer': 0,
        'params': 0,
        'fold_inner': 0,
        'epoch': 0,
        'model': None,
        'optimizer': None,
        'scheduler': None,

        'best_epoch': None,
        'best_epoch_val_loss': None,
        'current_patience': None,

        'best_epoch_counts': None,
        'inner_losses': None,
        'inner_accuracies': None,

        'best_inner_val_acc': None,
        'best_inner_val_loss': None,
        'best_inner_params': None,
        'average_best_epoch': None,
        'dict_all': None,

        'epoch_outer': 0,
        'model_outer': None,
        'optimizer_outer': None,
        'scheduler_outer': None,
        'train_outer': args.outer,

        'wandb_run_id': None,
    }

if args.wandb:
    wandb.login(key="d2fdeba96988784b072a586db7423ec7ce1266c2")
    if checkpoint['wandb_run_id'] is not None:
        wandb.init(config=args,
                   project=args.project,
                   name=args.csv_name,
                   dir=results_path,
                   job_type='training',
                   resume='must',
                   id=checkpoint['wandb_run_id'],
                   settings=wandb.Settings(init_timeout=300),)
    else:
        wandb.init(config=args,
                   project=args.project,
                   name=args.csv_name,
                   dir=results_path,
                   job_type='training',
                   reinit=True,
                   settings=wandb.Settings(init_timeout=300),)

skf_outer = StratifiedKFold(n_splits=args.n_fold, random_state=args.seed, shuffle=True)
logger_inner_train = Logger(args.n_fold, args.n_class, args=args)
logger_inner_val = Logger(args.n_fold, args.n_class, args=args)
logger_outer_train = Logger(args.n_fold, args.n_class, args=args)
logger_outer_test = Logger(args.n_fold, args.n_class, args=args)

for fold_outer, (train_val_idx, test_idx) in enumerate(skf_outer.split(dataset, dataset.labels)):
    if fold_outer < checkpoint['fold_outer']: continue
    train_val_dataset = torch.utils.data.Subset(dataset, train_val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    best_inner_val_acc = 0.0
    best_inner_val_loss = float('inf')
    best_inner_params = ParameterGrid(param_grid)[0]
    average_best_epoch = args.n_epoch
    if checkpoint['best_inner_val_acc'] is not None: best_inner_val_acc = checkpoint['best_inner_val_acc']
    if checkpoint['best_inner_val_loss'] is not None: best_inner_val_loss = checkpoint['best_inner_val_loss']
    if checkpoint['best_inner_params'] is not None: best_inner_params = checkpoint['best_inner_params']
    if checkpoint['average_best_epoch'] is not None: average_best_epoch = checkpoint['average_best_epoch']
    if checkpoint['dict_all'] is not None and args.outer: dict_all = checkpoint['dict_all']

    skf_inner = StratifiedKFold(n_splits=args.n_fold, random_state=args.seed, shuffle=True)

    for i, params in enumerate(ParameterGrid(param_grid)):
        if checkpoint['train_outer']: break
        if i < checkpoint['params']: continue
        best_epoch_counts = []
        inner_losses = []
        inner_accuracies = []
        if checkpoint['best_epoch_counts'] is not None: best_epoch_counts = checkpoint['best_epoch_counts']
        if checkpoint['inner_losses'] is not None: inner_losses = checkpoint['inner_losses']
        if checkpoint['inner_accuracies'] is not None: inner_accuracies = checkpoint['inner_accuracies']

        for fold_inner, (train_idx, val_idx) in enumerate(skf_inner.split(train_val_dataset, train_val_dataset.dataset.labels[train_val_idx])):
            if fold_inner < checkpoint['fold_inner']: continue
            train_dataset = torch.utils.data.Subset(train_val_dataset, train_idx)
            val_dataset = torch.utils.data.Subset(train_val_dataset, val_idx)
            original_train_idx = train_val_idx[train_idx]
            original_val_idx = train_val_idx[val_idx]

            best_epoch = args.n_epoch
            best_epoch_val_loss = float('inf')
            current_patience = 0
            if checkpoint['best_epoch'] is not None: best_epoch = checkpoint['best_epoch']
            if checkpoint['best_epoch_val_loss'] is not None: best_epoch_val_loss = checkpoint['best_epoch_val_loss']
            if checkpoint['current_patience'] is not None: current_patience = checkpoint['current_patience']

            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

            model = Ourmutimodel(args, params, priori).to(args.device)
            model = load_model(model, args, params)
            optimizer, scheduler = load_optimizer_scheduler(model, args, params, train_loader)
            # criterion = nn.CrossEntropyLoss(weight=weight.to(args.device))
            criterion = nn.CrossEntropyLoss()
            if checkpoint['model'] is not None: model.load_state_dict(checkpoint['model'])
            if checkpoint['optimizer'] is not None: optimizer.load_state_dict(checkpoint['optimizer'])
            if checkpoint['scheduler'] is not None: scheduler.load_state_dict(checkpoint['scheduler'])

            for epoch in range(checkpoint['epoch'], args.n_epoch):
                print(f'====')
                logger_inner_train.initialize(fold_inner)
                idxs = f'inner-train fold:{fold_outer}-({i})-{fold_inner} epoch:{epoch}'
                cur_train_loss, pred, prob, labels, _ = step(model=model,
                            dataloader=train_loader,
                            criterion=criterion,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            args=args,
                            params=params,
                            idxs=idxs)
                logger_inner_train.add(k=fold_inner, pred=pred, true=labels, prob=prob)
                # samples_train = logger_inner_train.get(fold_inner)
                metrics_inner_train = logger_inner_train.evaluate(fold_inner)
                cur_train_loss_formatted = {k: f'{v:.4f}' for k, v in cur_train_loss.items()}
                metrics_inner_train_formatted = {k: f'{v:.4f}' for k, v in metrics_inner_train.items()}
                print(f'loss:{cur_train_loss_formatted}', f'metrics:{metrics_inner_train_formatted}')
                if args.wandb:
                    wandb.log({
                        **{f'inner_{k}': v for k, v in cur_train_loss.items()},
                        **{f'inner_{k}': v for k, v in metrics_inner_train.items()},
                        'inner_epoch': epoch,
                    })

                logger_inner_val.initialize(fold_inner)
                idxs = f'inner-val fold:{fold_outer}-({i})-{fold_inner} epoch:{epoch}'
                cur_val_loss, pred, prob, labels, _ = step(model=model,
                            dataloader=val_loader,
                            criterion=criterion,
                            optimizer=None,
                            scheduler=None,
                            args=args,
                            params=params,
                            idxs=idxs)
                logger_inner_val.add(k=fold_inner, pred=pred, true=labels, prob=prob)
                # samples_train = logger_inner_val.get(fold_inner)
                metrics_inner_val = logger_inner_val.evaluate(fold_inner)
                cur_val_loss_formatted = {k: f'{v:.4f}' for k, v in cur_val_loss.items()}
                metrics_inner_val_formatted = {k: f'{v:.4f}' for k, v in metrics_inner_val.items()}
                print(f'loss:{cur_val_loss_formatted}', f'metrics:{metrics_inner_val_formatted}')
                if args.wandb:
                    wandb.log({
                        **{f'inner_{k}': v for k, v in cur_val_loss.items()},
                        **{f'inner_{k}': v for k, v in metrics_inner_val.items()},
                        'inner_epoch': epoch,
                    })

                if best_epoch_val_loss > cur_val_loss['loss'] or epoch < 1000:
                    current_patience = 0
                    best_epoch_val_loss = cur_val_loss['loss']
                    best_epoch = epoch + 1
                else:
                    current_patience += 1  # early stop
                    if current_patience > args.patience:
                        break

                # save checkpoint
                if args.savecheckpoint:
                    torch.save({
                        'fold_outer': fold_outer,
                        'params': i,
                        'fold_inner': fold_inner,
                        'epoch': epoch + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict() if scheduler is not None else None,

                        'best_epoch': best_epoch,
                        'best_epoch_val_loss': best_epoch_val_loss,
                        'current_patience': current_patience,

                        'best_epoch_counts': best_epoch_counts,
                        'inner_losses': inner_losses,
                        'inner_accuracies': inner_accuracies,

                        'best_inner_val_acc': best_inner_val_acc,
                        'best_inner_val_loss': best_inner_val_loss,
                        'best_inner_params': best_inner_params,
                        'average_best_epoch': average_best_epoch,
                        'dict_all': dict_all if not args.outer else None,

                        'epoch_outer': 0,
                        'model_outer': None,
                        'optimizer_outer': None,
                        'scheduler_outer': None,
                        'train_outer': False,

                        'wandb_run_id': wandb.run.id if args.wandb else None,  # save current wandb run ID
                        }, os.path.join(results_path, f'checkpoint.pth'))

            if best_epoch_val_loss != float('inf'):
                best_epoch_counts.append(best_epoch)
                inner_losses.append(best_epoch_val_loss.item())
                inner_accuracies.append(metrics_inner_val['accuracy'])
            print(best_epoch, best_epoch_val_loss)

            if args.wandb: wandb.log({'best_epoch': best_epoch, 'best_epoch_val_loss': best_epoch_val_loss})
            checkpoint.update({'epoch': 0, 'model': None, 'optimizer': None, 'scheduler': None,
                               'best_epoch': None, 'best_epoch_val_loss': None, 'current_patience': None})

        mean_inner_accuracy = np.mean(inner_accuracies)
        if mean_inner_accuracy > best_inner_val_acc:
            best_inner_val_acc = mean_inner_accuracy
            best_inner_params = params
            average_best_epoch = int(np.mean(best_epoch_counts))
        # mean_inner_loss = np.mean(inner_losses)
        # if mean_inner_loss < best_inner_val_loss:
        #     best_inner_val_loss = mean_inner_loss
        #     best_inner_params = params
        #     average_best_epoch = int(np.mean(best_epoch_counts))

    print(f"Best inner params: {best_inner_params}, \n"
          f"Best inner val acc: {best_inner_val_acc}, "
          f"Best inner val loss: {best_inner_val_loss}, "
          f"Best inner epoch: {average_best_epoch}")
    if args.wandb: wandb.log({'best_inner_params': best_inner_params,
                              'best_inner_val_acc': best_inner_val_acc,
                              'best_inner_val_loss': best_inner_val_loss,
                              'average_best_epoch': average_best_epoch})

    if not args.outer:
        dict_all['best_inner_params'].append(best_inner_params)
        dict_all['best_inner_val_acc'].append(best_inner_val_acc)
        dict_all['best_inner_val_loss'].append(best_inner_val_loss)
        dict_all['average_best_epoch'].append(average_best_epoch)
        np.save(os.path.join(results_path, f'dict_all.npy'), dict_all)

    # save checkpoint
    if args.savecheckpoint:
        torch.save({
            'fold_outer': fold_outer,
            'params': 0,
            'fold_inner': 0,
            'epoch': 0,
            'model': None,
            'optimizer': None,
            'scheduler': None,

            'best_epoch': None,
            'best_epoch_val_loss': None,
            'current_patience': None,

            'best_epoch_counts': None,
            'inner_losses': None,
            'inner_accuracies': None,

            'best_inner_val_acc': best_inner_val_acc,
            'best_inner_val_loss': best_inner_val_loss,
            'best_inner_params': best_inner_params,
            'average_best_epoch': average_best_epoch,
            'dict_all': dict_all if not args.outer else None,

            'epoch_outer': 0,
            'model_outer': None,
            'optimizer_outer': None,
            'scheduler_outer': None,
            'train_outer': True,

            'wandb_run_id': wandb.run.id if args.wandb else None,  # save current wandb run ID
        }, os.path.join(results_path, f'checkpoint.pth'))

    train_loader = DataLoader(train_val_dataset, batch_size=best_inner_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = Ourmutimodel(args, best_inner_params, priori).to(args.device)
    model = load_model(model, args, best_inner_params)
    optimizer, scheduler = load_optimizer_scheduler(model, args, best_inner_params, train_loader)
    # criterion = nn.CrossEntropyLoss(weight=weight.to(args.device))
    criterion = nn.CrossEntropyLoss()
    if checkpoint['model_outer'] is not None: model.load_state_dict(checkpoint['model_outer'])
    if checkpoint['optimizer_outer'] is not None: optimizer.load_state_dict(checkpoint['optimizer_outer'])
    if checkpoint['scheduler_outer'] is not None: scheduler.load_state_dict(checkpoint['scheduler_outer'])
    for epoch in range(checkpoint['epoch_outer'], average_best_epoch):
        print(f'====')
        logger_outer_train.initialize(fold_outer)
        idxs = f'outer-train fold:{fold_outer} epoch:{epoch}'
        cur_train_loss, pred, prob, labels, _ = step(model=model,
                              dataloader=train_loader,
                              criterion=criterion,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              args=args,
                              params=best_inner_params,
                              idxs=idxs)
        logger_outer_train.add(k=fold_outer, pred=pred, true=labels, prob=prob)
        metrics_outer_train = logger_outer_train.evaluate(fold_outer)
        cur_train_loss_formatted = {k: f'{v:.4f}' for k, v in cur_train_loss.items()}
        metrics_outer_train_formatted = {k: f'{v:.4f}' for k, v in metrics_outer_train.items()}
        # , model.alpha.data.item(), model.alpha.grad, model.alpha.requires_grad
        print(f'loss:{cur_train_loss_formatted}', f'metrics:{metrics_outer_train_formatted}')
        if args.wandb:
            wandb.log({
                **{f'outer_{k}': v for k, v in cur_train_loss.items()},
                **{f'outer_{k}': v for k, v in metrics_outer_train.items()},
                'outer_epoch': epoch,
            })

        logger_outer_test.initialize(fold_outer)
        idxs = f'outer-test fold:{fold_outer} epoch:{epoch}'
        cur_test_loss, pred, prob, labels, attn_test = step(model=model,
                            dataloader=test_loader,
                            criterion=criterion,
                            optimizer=None,
                            scheduler=None,
                            args=args,
                            params=best_inner_params,
                            idxs=idxs)
        logger_outer_test.add(k=fold_outer, pred=pred, true=labels, prob=prob)
        metrics_outer_test = logger_outer_test.evaluate(fold_outer)
        cur_test_loss_formatted = {k: f'{v:.4f}' for k, v in cur_test_loss.items()}
        metrics_outer_test_formatted = {k: f'{v:.4f}' for k, v in metrics_outer_test.items()}
        print(f'loss:{cur_test_loss_formatted}', f'metrics:{metrics_outer_test_formatted}')
        if args.wandb:
            wandb.log({
                **{f'outer_{k}': v for k, v in cur_test_loss.items()},
                **{f'outer_{k}': v for k, v in metrics_outer_test.items()},
                'outer_epoch': epoch,
            })

        # save checkpoint
        if args.savecheckpoint:
            torch.save({
                'fold_outer': fold_outer,
                'params': 0,
                'fold_inner': 0,
                'epoch': 0,
                'model': None,
                'optimizer': None,
                'scheduler': None,

                'best_epoch': None,
                'best_epoch_val_loss': None,
                'current_patience': None,

                'best_epoch_counts': None,
                'inner_losses': None,
                'inner_accuracies': None,

                'best_inner_val_acc': best_inner_val_acc,
                'best_inner_val_loss': best_inner_val_loss,
                'best_inner_params': best_inner_params,
                'average_best_epoch': average_best_epoch,
                'dict_all': dict_all if not args.outer else None,

                'epoch_outer': epoch + 1,
                'model_outer': model.state_dict(),
                'optimizer_outer': optimizer.state_dict(),
                'scheduler_outer': scheduler.state_dict() if scheduler is not None else None,
                'train_outer': True,

                'wandb_run_id': wandb.run.id if args.wandb else None,  # save current wandb run ID
            }, os.path.join(results_path, f'checkpoint.pth'))

    logger_outer_test.to_csv(results_path, fold_outer)
    if args.savefile:
        torch.save(model.state_dict(), os.path.join(results_path, f'model_{fold_outer}.pth'))
        np.save(os.path.join(results_path, f'attn_{fold_outer}.npy'), attn_test)
    # attn_all.append(attn_test)

    checkpoint.update({'params': 0, 'fold_inner': 0, 'epoch': 0, 'model': None, 'optimizer': None, 'scheduler': None,
                       'best_epoch': None, 'best_epoch_val_loss': None, 'current_patience': None,
                       'best_epoch_counts': None, 'inner_losses': None, 'inner_accuracies': None,
                       'best_inner_val_acc': None, 'best_inner_val_loss': None, 'best_inner_params': None, 'average_best_epoch': None,
                       'dict_all': dict_all if not args.outer else None,
                       'epoch_outer': 0, 'model_outer': None, 'optimizer_outer': None, 'scheduler_outer': None, 'train_outer': args.outer})
    # sys.exit(0)
logger_outer_test.to_csv(results_path)
if args.savefile: torch.save(logger_outer_test.get(), os.path.join(results_path, f'samples.pkl'))
# np.save(attn_all, os.path.join(results_path, f'attn_{args.csv_name}.npy'))

if os.path.isfile(os.path.join(results_path, f'checkpoint.pth')):
    print('delete checkpoint file')
    os.remove(os.path.join(results_path, f'checkpoint.pth'))
