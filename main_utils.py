import copy
from genericpath import exists

import torch.nn as nn
import numpy as np
import datetime
import torch
import math
import os

eps = 1e-8


def log_message(logger, message):
    if logger is None:
        print(message)
    else:
        logger.info(message)


def get_path_norm(grouped_layer, w_norm_deg=2, v_norm_deg=1, requires_grad=False):
    # :param grouped_layer: two torch.nn.Modules.
    # path norm
    w = grouped_layer[0].weight if requires_grad else grouped_layer[0].weight.data
    v = grouped_layer[1].weight if requires_grad else grouped_layer[1].weight.data

    if isinstance(grouped_layer[0], nn.Linear) and isinstance(grouped_layer[1], nn.Linear):
        path_norm = torch.linalg.vector_norm(w, dim=1, ord=w_norm_deg) * \
                    torch.linalg.vector_norm(v, dim=0, ord=v_norm_deg)
    elif isinstance(grouped_layer[0], nn.Conv2d) and isinstance(grouped_layer[1], nn.Conv2d):
        path_norm = torch.linalg.vector_norm(w, dim=(1, 2, 3), ord=w_norm_deg) * \
                    torch.linalg.vector_norm(v, dim=(0, 2, 3), ord=v_norm_deg)
    else:
        raise ValueError("Wrong layers passed into path norm implementation.")

    return path_norm


def get_gradient_norm(model, device, criterion, data_loader):
    model.eval()
    gradient_norm_max = -1.
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad_()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        dim_tuple = np.arange(len(data.shape)).tolist()[1:]
        gradient_norm = torch.linalg.vector_norm(data.grad, ord=2, dim=dim_tuple).max()
        if gradient_norm_max < gradient_norm:
            gradient_norm_max = gradient_norm

    return gradient_norm_max


def update_iter_and_epochs(dataset, args, logger):
    per_epoch_iter = math.ceil(len(dataset.train_loader.dataset) // args.batch_size)  # number of iterations per epoch
    if args.total_epoch == 0:
        args.total_epoch = args.total_iter // per_epoch_iter + 1
    elif args.total_iter == 0:
        args.total_iter = args.total_epoch * per_epoch_iter
    else:
        if not args.lr_rewind:
            assert args.total_iter == args.total_epoch * per_epoch_iter
        else:
            logger.info("Since it is learning rate rewinding, use self-defined total_iter and total_epochs")
    logger.info("Update the total iter to be {}, and total epoch to be {}".format(args.total_iter, args.total_epoch))


def get_dataset(args, logger):
    import dataset
    message = "=> Getting {} dataset".format(args.which_dataset)
    log_message(logger, message)
    dataset = getattr(dataset, args.which_dataset)(args)
    return dataset


def get_criterion(criterion_type):
    if criterion_type.lower() == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif criterion_type.lower() == 'mse':
        criterion = torch.nn.MSELoss(reduction='mean')
    elif criterion_type.lower() == 'bce':
        criterion = torch.nn.BCELoss(reduction='mean')
    else:
        raise NotImplementedError("only support criterion CE (cross entropy) | MSE (mean squared error) | BCE (Binary Cross Entropy)")
    return criterion


def get_model(args, logger, dataset):
    import models
    message = "=> Creating model {}".format(args.arch)
    log_message(logger, message)
    if args.arch.lower() in ["resmlp"]:
        if args.act_fn.lower() == 'relu':
            act_fn = nn.ReLU
        elif args.act_fn.lower() == 'gelu':
            act_fn = nn.GELU
        else:
            raise NotImplementedError("We only support ReLU or GELU, please double check your args.act_fn")
        model = models.__dict__[args.arch](
            input_dim=dataset.input_dim, num_classes=dataset.num_classes,
            drop_path_rate=args.drop_path, act_layer=act_fn,
            pretrained=args.load_pretrained_model,
            two_layer_classifier=args.two_layer_classifier,
            algo=args.algo,
            w_norm_deg=args.w_norm_degree,
            v_norm_deg=args.v_norm_degree)
    elif "pf" in args.arch.lower():
        model = models.__dict__[args.arch](input_dim=dataset.input_dim, num_classes=dataset.num_classes,
                                           LR_FACTOR=args.pf_lr_factor)
    elif "lenet_small" in args.arch.lower():
        model = models.__dict__[args.arch](input_dim=dataset.input_dim, num_classes=dataset.num_classes,
                                           nact1=args.small_network_nact1, nact2=args.small_network_nact2,
                                           algo=args.algo)
    elif "lenet" in args.arch.lower():
        model = models.__dict__[args.arch](input_dim=dataset.input_dim, num_classes=dataset.num_classes, algo=args.algo)
    elif ("shallow" in args.arch.lower()) or ("deep" in args.arch.lower()):
        model = models.__dict__[args.arch](input_dim=dataset.input_dim, input_channel=dataset.input_channel,
                                           num_hidden=args.num_hidden, num_classes=dataset.num_classes)
    elif "vgg" in args.arch.lower():
        model = models.__dict__[args.arch](num_classes=dataset.num_classes)
    else:
        model = models.__dict__[args.arch](input_dim=dataset.input_dim, num_classes=dataset.num_classes)
    
    if args.load_pretrained_model:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model


def get_optimizer(args, model):
    opt_algo = args.optimizer
    lr = args.lr
    wd = args.wd
    mom = args.momentum
    if opt_algo.lower() == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    elif opt_algo.lower() == "sgd":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=mom, weight_decay=wd)
    elif opt_algo.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    else:
        raise NotImplementedError("Only support Adam, AdamW and SGD")
    
    if args.load_pretrained_model:
        checkpoint = torch.load(args.checkpoint_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return optimizer


def get_scheduler(optimizer, logger, args):
    scheduler = args.lr_scheduler
    max_epochs = args.total_epoch
    if scheduler == 'cosine_lr':
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=max_epochs,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            k_decay=getattr(args, 'lr_k_decay', 1.0),
        )
        message = "scheduler: use cosine learning rate decay, with max epochs {}".format(max_epochs)
    elif scheduler == "multi_step":
        gamma = args.gamma
        milestones = args.milestone
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        message = "scheduler: use multistep learning rate decay, with milestones {} and gamma {}".format(milestones, gamma)

    else:
        message = "Policy not specified. Default is None"
        lr_scheduler = None
    log_message(logger, message)

    return lr_scheduler


def set_seed(seed, logger):
    import random
    import os
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # making sure GPU runs are deterministic even if they are slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    message = "Seeded everything: {}".format(seed)
    log_message(logger, message)


def set_dest_dir(args):
    subfolder_path = "results/{}_{}".format(args.which_dataset, args.arch)
    os.makedirs(subfolder_path, exist_ok=True)
    now = datetime.datetime.now().strftime('%m%d%H%M%S')
    if args.load_pretrained_model:
        dest_dir = args.results_dir
    else:
        dest_dir = os.path.join(subfolder_path, "{}_{}".format(now, args.logger_name))
    os.makedirs(dest_dir, exist_ok=True)
    args.dest_dir = dest_dir


def test(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


def calc_and_print_nonzeros_neuron(model, w_norm_deg, v_norm_deg, logger, thr_max=0.0001):
    nz, tot = 0, 0
    grouped_pn = []
    total_pn = 0.
    grouped_active = []
    grouped_total = []
    manually_calc_mask = []
    # logger.info("start checking if the mask and zeros in model are matched")
    for idx, grouped_layer in enumerate(model.grouped_layers):
        if model.algo == 'v0':
            pn = get_path_norm(grouped_layer, w_norm_deg, v_norm_deg)
        else:
            pn = model.scale_layers[idx].data
        max_neuron_pn = torch.max(pn)
        N = pn.shape[0]
        mask = (pn > 0)
        manually_calc_mask.append(mask)
        grouped_pn.append(pn)
        total_pn += pn.sum().item()
        #grouped_active.append((pn > 0).sum())
        grouped_active.append((pn > thr_max * max_neuron_pn).sum())
        grouped_total.append(pn.flatten().shape[0])
        # logger.info("group {}: model: {} / {} remain".format(idx, (pn.abs() > 0).sum(), N))
        #nz += (pn > 0).sum()
        nz += (pn > thr_max * max_neuron_pn).sum()
        tot += N
    return nz / (tot + eps) * 100., total_pn, grouped_pn, grouped_active, grouped_total, manually_calc_mask


def calc_and_print_nonzeros_weight(model, logger):
    nonzero = 0
    total = 0
    nonzero_params = []  # list of nonzero parameters (each entry is a layer)
    total_params = []  # list of total parameters (each entry is a layer)
    l2_norm = 0.
    for idx, grouped_layer in enumerate(model.grouped_layers):
        tmp_nz = []
        tmp_total = []
        for layer_id in [0, 1]:
            tensor = grouped_layer[layer_id].weight.data
            l2_norm_tensor = torch.pow(tensor, 2).sum()
            l2_norm += l2_norm_tensor
            nz_count = torch.nonzero(tensor).shape[0]
            tmp_nz.append(nz_count)
            tot_params = tensor.numel()
            tmp_total.append(tot_params)
            nonzero += nz_count
            total += tot_params
            # logger.info(f'group {idx}-{layer_id} | nonzeros = {nz_count:7} / {tot_params:7} ({100 * nz_count / tot_params:6.2f}%) | total_pruned = {tot_params - nz_count :7} | shape = {tensor.shape}')
        nonzero_params.append(tmp_nz)
        total_params.append(tmp_total)
    # logger.info(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, ({100 * nonzero / total:6.2f}% remained)')
    return nonzero / (total + eps) * 100., l2_norm, nonzero_params, total_params


def put_mask_on(model, mask, algo, flag_unstructure=False):
    with torch.no_grad():
        if flag_unstructure:
            for name, param in model.named_parameters():
                if name in model.prunable_layer_name:
                    param.data = param.data * mask[name]
        else:
            for idx, grouped_layer in enumerate(model.grouped_layers):
                if isinstance(grouped_layer[0], nn.Linear) and isinstance(grouped_layer[1], nn.Linear):
                    w = grouped_layer[0].weight.data * mask[idx][:, None]
                    v = grouped_layer[1].weight.data * mask[idx]
                elif isinstance(grouped_layer[0], nn.Conv2d) and isinstance(grouped_layer[1], nn.Conv2d):
                    w = grouped_layer[0].weight.data * mask[idx][:, None, None, None]
                    v = grouped_layer[1].weight.data * mask[idx][:, None, None]
                w_b = grouped_layer[0].bias.data * mask[idx]
                grouped_layer[0].weight.data = w.data
                grouped_layer[0].bias.data = w_b.data
                grouped_layer[1].weight.data = v.data
            if algo == "v2":
                for idx, layer in enumerate(model.scale_layers):
                    s = layer.data * mask[idx]
                    layer.data = s.data


def generate_random_mask(model, mask, device, target_sparsity):
    """
    :param model: the initialized model
    :param mask: the initialized mask
    :param device:
    :param target_sparsity:
    """
    for idx, N in enumerate(model.num_neurons):
        K = int(N * target_sparsity / 100.0)  # number of active neurons
        tmp_array = np.array([0] * (N - K) + [1] * K)  # [N,]
        np.random.shuffle(tmp_array)  # [N,]
        idx_mask = torch.from_numpy(tmp_array).float()  # [N,]
        mask[idx] = idx_mask.to(device)
    return mask


def freeze_previous_layer(model):
    for param in model.parameters():
        if param.requires_grad:
            param.requires_grad = False
    if isinstance(model.head, nn.Linear):
        model.head.weight.requires_grad = True
        model.head.bias.requires_grad = True
    elif isinstance(model.head, nn.Sequential):
        model.head[0].weight.requires_grad = True
        model.head[0].bias.requires_grad = True
        model.head[2].weight.requires_grad = True
        model.head[2].bias.requires_grad = True
    else:
        raise NotImplementedError


def convert_v2_to_v0(model):
    assert model.algo == "v2"

    model2 = copy.deepcopy(model)

    for idx, grouped_layer in enumerate(model2.grouped_layers):
        grouped_layer[1].weight.data *= model2.scale_layers[idx][None, :]

    model2.algo = "v0"
    return model2

def out_sparsity(model, active_neurons):
    v = model.linear2.weight.detach()
    out_weights = v[:,active_neurons]
    sparsity = torch.max(torch.abs(out_weights), dim=0)[0] / torch.linalg.norm(out_weights,ord=1, dim=0)
    spars_perc = torch.sum(sparsity) / len(sparsity)
    return spars_perc


