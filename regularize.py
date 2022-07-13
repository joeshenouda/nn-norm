import torch.nn as nn
import numpy as np
import torch
import json
import tqdm
import time
import os

from main_utils import get_path_norm, test, calc_and_print_nonzeros_neuron, calc_and_print_nonzeros_weight, get_gradient_norm, out_sparsity, get_out_weights
from prune_algo import prune_or_regularize, layerwise_balance, normalize_w, collect_other_l2_norm, collect_grouped_norm


def test_and_log(model, args, dataset, criterion, device, result_dict, w_norm_deg, v_norm_deg, logger):
    if args.which_dataset.lower() == "rnnl":
        # dataset
        train_loader = dataset.train_loader
        test_loader = dataset.test_loader
        test_acc = test(model, args, criterion, test_loader, device)
        result_dict['loss']['test'].append(test_acc)
    else:
        train_loader = dataset.train_loader
        val_loader = dataset.val_loader
        test_loader = dataset.test_loader

    if args.which_dataset.lower() == "rnnl" or args.which_dataset.lower() == "mnist_binary_synth" or args.which_dataset.lower() == "mnist_subset_synth":
        val_acc = 0
        train_acc = 0
    else:
        test_acc = test(model, args, criterion, test_loader, device)
        result_dict['acc']['test'].append(test_acc)
        val_acc = test(model, args, criterion, val_loader, device)
        result_dict['acc']['val'].append(val_acc)
        train_acc = test(model, args, criterion, train_loader, device)
        result_dict['acc']['train'].append(train_acc)

    # gradient norm
    #gradient_norm_max = get_gradient_norm(model, device, criterion, val_loader)

    # calculate sparsity
    nsp, total_pn, pns, nacts, totals, _ = calc_and_print_nonzeros_neuron(model, w_norm_deg, v_norm_deg, logger, thr_max=0)
    result_dict['act']['nact'].append(nacts)
    result_dict['act']['nact_total'].append(totals)
    result_dict['act']['pns'].append(pns)
    wsp, l2_norm, wacts, totals = calc_and_print_nonzeros_weight(model, logger)
    result_dict['act']['wact'].append(wacts)
    result_dict['act']['wact_total'].append(totals)
    
    max_pn = torch.max(pns[0])
    #thr_max=0.0001
    thr_max = 0
    idx_act_neurons = pns[0] > thr_max * max_pn
    out_sparse_perc = out_sparsity(model, idx_act_neurons)
    result_dict['act']['out_sparse'] .append(out_sparse_perc)

    wandb_dict = dict({
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "wact": wsp,
        "pn_list" : pns,
        "nact": nacts[0],
        "l2_norm": l2_norm,
        #"grad_norm_max": gradient_norm_max,
        "path_norm_total": total_pn,
        "out_sparse": out_sparse_perc
    })

    for idx, item in enumerate(pns):
        wandb_dict["mean_path_norm_group_{}".format(idx + 1)] = item.mean().item()
        wandb_dict["path_norm_min_group_{}".format(idx + 1)] = torch.min(item[torch.nonzero(item)])
        wandb_dict["path_norm_max_group_{}".format(idx + 1)] = torch.max(item[torch.nonzero(item)])

    result_dict['loss']['L2_norm'].append(l2_norm)
    _, total_pn, _, _, _, _ = calc_and_print_nonzeros_neuron(model, 2, 2, logger)
    result_dict['loss']['path_norm22'].append(total_pn)
    wandb_dict['path_norm22'] = total_pn
    _, total_pn, _, _, _, _ = calc_and_print_nonzeros_neuron(model, 2, 1, logger)
    result_dict['loss']['path_norm21'].append(total_pn)
    wandb_dict['path_norm21'] = total_pn
    #result_dict['loss']['grad_norm_max'].append(gradient_norm_max)

    return result_dict, wandb_dict


def wandb_log(wandb, wandb_dict, args, model, idx_iter, epoch, train_loss, optimizer):
    wandb_dict['idx_iter'] = idx_iter
    wandb_dict['train_loss'] = train_loss

    for idx, group_pn in enumerate(wandb_dict['pn_list']):
        pn_data = [[pn.item()] for pn in group_pn]
        pn_table = wandb.Table(data = pn_data, columns = ["Path Norms Group {}".format(idx+1)])
        wandb_dict["path_norm_hist_group_{}".format(idx+1)] = wandb.plot.histogram(pn_table, "Path Norms Group {}".format(idx+1), title="Distribution of Path Norms")

        out_weights = get_out_weights(model, group_pn)
        out_weights_list = out_weights.cpu().numpy().tolist()
        #import ipdb; ipdb.set_trace()
        out_weight_table = wandb.Table(data = out_weights_list, columns=["v{}".format(i) for i in range(out_weights.shape[1])])
        wandb_dict["out_weights_table"] = out_weight_table

    w_norm_max, v_norm_max, w_norm_min, v_norm_min = -1, -1, 100, 100
    for grouped_layer in model.grouped_layers:
        w = grouped_layer[0].weight.detach()
        v = grouped_layer[1].weight.detach()
        w_norm = torch.linalg.vector_norm(w, dim=1, ord=2)
        v_norm = torch.linalg.vector_norm(v, dim=0, ord=1)
        if w_norm.max() > w_norm_max:
            w_norm_max = w_norm.max()
        if w_norm.min() < w_norm_min:
            w_norm_min = w_norm[torch.nonzero(w_norm)].min()
        if v_norm.max() > v_norm_max:
            v_norm_max = v_norm.max()
        if v_norm.min() < v_norm_min:
            v_norm_min = v_norm[torch.nonzero(v_norm)].min()
    wandb_dict['w_norm_max'] = w_norm_max
    wandb_dict['w_norm_min'] = w_norm_min
    wandb_dict['v_norm_max'] = v_norm_max
    wandb_dict['v_norm_min'] = v_norm_min
    wandb_dict['lr'] = optimizer.param_groups[0]["lr"]
    if wandb is not None:
        wandb.log(wandb_dict, step=epoch)


def init_result_dict(args, dataset, wandb):
    if args.load_pretrained_model:
        PATH_results = os.path.join(args.dest_dir, "result.pt")
        result_dict = torch.load(PATH_results)
    else:
        acc_dict, act_dict, loss_dict = {}, {}, {}
        acc_dict['train'], acc_dict['test'], acc_dict['val'] = [], [], []
        act_dict['nact'], act_dict['wact'], act_dict['nact_total'], act_dict['wact_total'], act_dict['pns'], act_dict['out_sparse'] = [], [], [], [], [], []
        loss_dict['loss'], loss_dict['test'], loss_dict['path_norm22'], loss_dict['path_norm21'], loss_dict['L2_norm'], loss_dict['grad_norm_max'] = [], [], [], [], [], []
        result_dict = {'acc': acc_dict, 'act': act_dict, 'loss': loss_dict, 'wandb_id': wandb.run.id}
        if args.which_dataset.lower() == 'rnnl':
            rnnl_net = dataset.rnnl_net
            rnnl_net_path_norm21 = get_path_norm(rnnl_net.grouped_layers[0], w_norm_deg=2, v_norm_deg=1)
            rnnl_net_path_norm22 = get_path_norm(rnnl_net.grouped_layers[0], w_norm_deg=2, v_norm_deg=2)
            result_dict['rnnl_net_path_norm21'] = rnnl_net_path_norm21
            result_dict['rnnl_net_path_norm22'] = rnnl_net_path_norm22
    return result_dict


def trainer(dataset, device, model, args, optimizer, scheduler, criterion, logger, wandb):
    """
    this updated version calculate path-norm as ||w_k||_2 ||v_k||_1
    """

    # use args:
    algo = args.algo
    total_iter = args.total_iter
    total_epoch = args.total_epoch
    log_freq = args.log_freq
    save_freq = args.save_freq
    thr = args.threshold
    dest_dir = args.dest_dir
    w_norm_deg = args.w_norm_degree
    v_norm_deg = args.v_norm_degree
    flag_with_loss_term = args.with_loss_term
    loss_term = args.loss_term
    flag_with_prox_upd = args.with_prox_upd
    flag_layerwise_balance = args.layerwise_balance
    plus = args.plus

    iter_period = args.iter_period
    prune_iter = args.prune_iter
    debias_iter = args.debias_iter

    assert args.wd == 0, "weight decay needs to be set to zero to ensure "
    assert algo == 'v0', "Let's only work on algorithm v0 structure for now"
    result_dict = init_result_dict(args, dataset, wandb)

    if args.which_dataset.lower() == 'rnnl':
        print('Random NN Path Norm L2-L1: {}'.format(result_dict['rnnl_net_path_norm21']))
        print('Random NN Path Norm L2-L2: {}'.format(result_dict['rnnl_net_path_norm22']))

    # start training
    idx_iter = 0
    flag_iter = False  # flag of whether it has reached the total number of iterations
    
    start_epoch = 0
    start_iter = 0
    #import ipdb; ipdb.set_trace()

    if args.load_pretrained_model:
        checkpoint = torch.load(args.checkpoint_path)
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint['iter']
        last_loss = checkpoint['loss']
        idx_iter += start_iter+1
        total_iter += idx_iter
    #import ipdb; ipdb.set_trace()
    # begin training
    for idx_epoch in tqdm.tqdm(range(start_epoch, start_epoch+total_epoch)):
        #import ipdb; ipdb.set_trace()
        if flag_iter:  # has reached the total number of iterations
            break
        else:
            # Training
            model.train()
            for batch_idx, (imgs, targets) in enumerate(dataset.train_loader):
                actual_thr = thr * optimizer.param_groups[0]["lr"]
                imgs, targets = imgs.to(device), targets.to(device)

                output = model(imgs)

                # make labels one-hot if using MSELoss
                if args.criterion.lower() == 'mse' and args.which_dataset.lower() == 'mnist_binary':
                    bs = targets.shape[0]
                    out_dim = output.shape[1]
                    one_hot_targets = torch.zeros(bs,out_dim)
                    for i, targ in enumerate(one_hot_targets):
                        if args.flip_one_hot:
                            targ[1-targets[i]] = 1
                        else:
                            targ[targets[i]] = 1
                    targets = one_hot_targets.to(device)
                
                train_loss = criterion(output, targets)
                l2_norm_reg = collect_other_l2_norm(model, thr)
                if flag_with_loss_term and prune_iter < idx_iter % iter_period <= debias_iter:
                    other_norm_reg = collect_grouped_norm(model, thr, loss_term, w_norm_deg, v_norm_deg, plus)
                else:
                    other_norm_reg = 0.
                optimizer.zero_grad()
                (train_loss + l2_norm_reg + other_norm_reg).backward()
                optimizer.step()
                if flag_with_prox_upd:
                    model = normalize_w(model, algo, w_norm_deg)
                    model = layerwise_balance(model, algo, w_norm_deg, v_norm_deg)
                    if prune_iter < idx_iter % iter_period <= debias_iter:
                        model, _ = prune_or_regularize(model, actual_thr, algo, v_norm_deg)
                elif flag_layerwise_balance:
                    model = layerwise_balance(model, algo, w_norm_deg, v_norm_deg)

                # Frequency for Testing
                if idx_iter % log_freq == 0:
                    result_dict, wandb_dict = test_and_log(
                        model, args, dataset, criterion, device, result_dict, w_norm_deg, v_norm_deg, logger)
                    result_dict['loss']['loss'].append(train_loss.item())

                    PATH_result = os.path.join(dest_dir, "result.pt")
                    torch.save(result_dict, PATH_result)
                    wandb_log(wandb, wandb_dict, args, model, idx_iter, idx_epoch, train_loss, optimizer)
                    logger.info("Iter: {}, Loss: {:.5f}".format(idx_iter, train_loss.item()))
                    logger.info("Iter: {}, Active Nerons:{}".format(idx_iter, result_dict['act']['nact'][-1][0].item()))
                    logger.info("Iter: {}, Path Norm 22: {}".format(idx_iter, result_dict['loss']['path_norm22'][-1]))
                    logger.info("Iter: {}, Path Norm 21: {}".format(idx_iter, result_dict['loss']['path_norm21'][-1]))
                if idx_iter % save_freq == 0:
                    PATH_model = os.path.join(dest_dir, "model_idx_{}_acc_{}_sp_{}".format(
                        idx_iter, round(wandb_dict['test_acc'], 2), int(wandb_dict['nact'])).replace(".", "_") + ".pt")
                    torch.save({
                        'iter' : idx_iter,
                        'epoch' : idx_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss' : train_loss
                    }, PATH_model)
                    wandb.save(PATH_model)

                if idx_iter == total_iter:
                    flag_iter = True
                    break
                
                idx_iter += 1


            if scheduler is not None:
                # scheduler.step(idx_epoch)
                scheduler.step()

    return model, result_dict
