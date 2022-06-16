import torch
import torch.nn as nn
import torch.nn.functional as F
from main_utils import get_path_norm

eps = 1e-12


def collect_other_l2_norm(model, lambd):
    total_l2_norm = 0.
    for layer in model.other_layers:
        this_l2_norm = torch.pow(layer.weight, 2).sum()
        total_l2_norm += this_l2_norm
    return 0.5 * lambd * total_l2_norm


def collect_grouped_norm(model, lambd, loss_term, w_norm_deg, v_norm_deg, plus):
    total = 0.
    if loss_term == 'wd':
        for grouped_layer in model.grouped_layers:
            w = grouped_layer[0]
            w_l2_norm = torch.pow(w.weight, 2).sum()
            v = grouped_layer[1]
            v_l2_norm = torch.pow(v.weight, 2).sum()
            total += 0.5 * (w_l2_norm + v_l2_norm)
    elif loss_term == 'pn':
        for grouped_layer in model.grouped_layers:
            pn = get_path_norm(grouped_layer, w_norm_deg, v_norm_deg, requires_grad=True)
            if plus:
                w = grouped_layer[0].weight
                v = grouped_layer[1].weight
                pn = torch.pow(torch.linalg.vector_norm(w, dim=1, ord=w_norm_deg), 2).sum() + torch.pow(
                    torch.linalg.vector_norm(v, dim=0, ord=v_norm_deg), 2).sum()
                pn *= 0.5
            total += pn.sum()
    else:
        raise NotImplementedError("Only support loss term wd | pn")
    return lambd * total


def prune_or_regularize(model, thr, algo, v_norm_deg, mask=None, mask_null=None):
    if algo == 'v0':
        model, new_mask = prune_or_regularize_v0(mask, mask_null, model, thr, v_norm_deg)
    elif algo == 'v2':
        model, new_mask = prune_or_regularize_v2(mask, mask_null, model, thr, v_norm_deg)
    else:
        raise NotImplementedError
    return model, new_mask


def gmean(input_x, dim):
    input_x = torch.clip(input_x, min=eps)
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))


def prox_grad_upd_v0(w_k, v_k, lam, v_norm_deg):
    """
    This function does proximal gradient update for algorithm v0
    :param w_k: shape [N, in_dim]
    :param v_k: shape [out_dim, N]
    """
    if v_norm_deg == 1:
        v_k_upd = F.relu(torch.abs(v_k) - lam) * torch.sign(v_k)  # [out_dim, N]
    elif v_norm_deg == 2:
        v_k_norm = torch.linalg.vector_norm(v_k, dim=0, ord=2)  # [N,]
        v_k_upd = torch.where(v_k_norm <= lam, torch.zeros_like(v_k),
                              v_k - lam * v_k / torch.clip(v_k_norm[None, :], min=eps))  # [out_dim, N]
    else:
        raise NotImplementedError("We only implemented v norm degree of 1 and 2")

    return w_k, v_k_upd


def prox_grad_upd_v0_conv(w_k, v_k, lam, v_norm_deg):
    """
    This function does proximal gradient update for algorithm v0
    :param w_k: shape [N, in_dim, kernel, kernel]
    :param v_k: shape [out_dim, N, kernel, kernel]
    """
    assert len(w_k.shape) == 4 and len(v_k.shape) == 4, "Weight dimension has to be 4 for convolutional prox " \
                                                        "update v0. Got %d dimensions for w_k and %d dimensions for" \
                                                        " v_k" % (len(w_k.shape), len(v_k.shape))
    if v_norm_deg == 1:
        v_k_upd = F.relu(torch.abs(v_k) - lam) * torch.sign(v_k)  # [out_dim, N, K, K]
    elif v_norm_deg == 2:
        v_k_norm = torch.linalg.vector_norm(v_k, dim=(0, 2, 3), ord=2)  # [N,]
        v_k_upd = torch.where((v_k_norm <= lam)[:, None, None], torch.zeros_like(v_k),
                              v_k - lam * v_k / torch.clip(v_k_norm[:, None, None], min=eps))  # [out_dim, N, K, K]
    else:
        raise NotImplementedError("We only implemented v norm degree of 1 and 2")

    return w_k, v_k_upd


def prox_grad_upd_v2(s_k, lam):
    """
    :param s_k: shape [N,]
    """
    # layer normalization
    s_k_upd = F.relu(torch.abs(s_k) - lam)  # [N,]
    return s_k_upd


"""
def layerwise_balance(model, algo='v0', w_norm_deg=2, v_norm_deg=1):
    if algo == "v2":
        model.normalize(renormalize=True, w_norm_deg=w_norm_deg, v_norm_deg=v_norm_deg)
        pn_list = [item.data.sum() for item in model.scale_layers]
    else:
        pn_list = []
        for grouped_layer in model.grouped_layers:
            pn = get_path_norm(grouped_layer, w_norm_deg, v_norm_deg)
            pn_list.append(pn.sum())  # changed because of numerical issue: if sum, then the path norm will explode

    pn_array = torch.tensor(pn_list)  # [n_group]
    pn_mean = gmean(pn_array, dim=0)  # []
    if algo == "v2":
        for idx, s in enumerate(model.scale_layers):
            tmp = torch.clip(pn_list[idx], min=eps)
            scale = pn_mean / tmp
            s.data *= scale
    else:
        for idx, grouped_layer in enumerate(model.grouped_layers):
            tmp = torch.clip(pn_list[idx], min=eps)
            scale = pn_mean / tmp
            w = grouped_layer[0].weight.data  # [N, input_dim]
            w_b = grouped_layer[0].bias.data  # [N,]
            v = grouped_layer[1].weight.data  # [output_dim, N]
            v_b = grouped_layer[1].bias.data  # [output_dim,]
            if isinstance(grouped_layer[0], nn.Linear) and isinstance(grouped_layer[1], nn.Linear):
                w_norm = torch.linalg.vector_norm(w, dim=1, ord=w_norm_deg)  # [N,]
                w_norm = torch.clip(w_norm, min=eps)
                w_upd = w / w_norm[:, None]
                # v_upd = v * scale * w_norm[None, :]
                v_upd = v * scale
            elif isinstance(grouped_layer[0], nn.Conv2d) and isinstance(grouped_layer[1], nn.Conv2d):
                w_norm = torch.linalg.vector_norm(w, dim=(1, 2, 3), ord=w_norm_deg)  # [N,]
                w_norm = torch.clip(w_norm, min=eps)
                w_upd = w / w_norm[:, None, None, None]
                # v_upd = v * scale * w_norm[:, None, None]
                v_upd = v * scale
            w_b_upd = w_b / w_norm
            v_b_upd = v_b * scale
            grouped_layer[0].weight.data = w_upd.data
            grouped_layer[0].bias.data = w_b_upd.data
            grouped_layer[1].weight.data = v_upd.data
            grouped_layer[1].bias.data = v_b_upd.data
    return model
"""


def layerwise_balance(model, algo='v0', w_norm_deg=2, v_norm_deg=1):
    if algo == "v2":
        model.normalize(renormalize=True, w_norm_deg=w_norm_deg, v_norm_deg=v_norm_deg)
        pn_list = [item.data.sum() for item in model.scale_layers]
    else:
        pn_list = []
        for grouped_layer in model.grouped_layers:
            pn = get_path_norm(grouped_layer, w_norm_deg, v_norm_deg)
            pn_list.append(pn.sum())  # changed because of numerical issue: if sum, then the path norm will explode

    pn_array = torch.tensor(pn_list)  # [n_group]
    pn_mean = gmean(pn_array, dim=0)  # []
    if algo == "v2":
        for idx, s in enumerate(model.scale_layers):
            tmp = torch.clip(pn_list[idx], min=eps)
            scale = pn_mean / tmp
            s.data *= scale
    else:
        for idx, grouped_layer in enumerate(model.grouped_layers):
            tmp = torch.clip(pn_list[idx], min=eps)
            scale = pn_mean / tmp
            v = grouped_layer[1].weight.data  # [output_dim, N]
            v_b = grouped_layer[1].bias.data  # [output_dim,]
            v_upd = v * scale
            v_b_upd = v_b * scale
            grouped_layer[1].weight.data = v_upd.data
            grouped_layer[1].bias.data = v_b_upd.data
    return model


def normalize_w(model, algo='v0', w_norm_deg=2):
    assert w_norm_deg == 2, "Currently the projection only works for 2-norm"
    if algo == 'v0':
        for idx, grouped_layer in enumerate(model.grouped_layers):
            w = grouped_layer[0].weight.data  # [N, input_dim]
            w_b = grouped_layer[0].bias.data  # [N,]
            if isinstance(grouped_layer[0], nn.Linear) and isinstance(grouped_layer[1], nn.Linear):
                w_norm = torch.linalg.vector_norm(w, dim=1, ord=w_norm_deg)  # [N,]
                w_norm = torch.clip(w_norm, min=eps)
                w_upd = w / w_norm[:, None]
                # v_upd = v * w_norm[None, :]
            elif isinstance(grouped_layer[0], nn.Conv2d) and isinstance(grouped_layer[1], nn.Conv2d):
                w_norm = torch.linalg.vector_norm(w, dim=(1, 2, 3), ord=w_norm_deg)  # [N,]
                w_norm = torch.clip(w_norm, min=eps)
                w_upd = w / w_norm[:, None, None, None]
                # v_upd = v * w_norm[:, None, None]
            w_b_upd = w_b / w_norm

            grouped_layer[0].weight.data = w_upd.data
            grouped_layer[0].bias.data = w_b_upd.data
    return model


def prune_or_regularize_v0(mask, mask_null, model, thr, v_norm_deg):
    """This function prune the model by its magnitude， basically is what soft-thresholding does
    :param mask: the pervious mask
    :param model: the model to be pruned
    :param thr: if below the threshold, just set it to be zero
    :param w_norm_deg: the degree of norm for w
    :param v_norm_deg: the degree of norm for v
    :returns:
        model: the updated model (after apply the new mask)
        new_masks: the updated mask
    """

    new_masks = []
    for idx, grouped_layer in enumerate(model.grouped_layers):
        w = grouped_layer[0].weight.data
        v = grouped_layer[1].weight.data
        if isinstance(grouped_layer[0], nn.Linear) and isinstance(grouped_layer[1], nn.Linear):
            w_upd, v_upd = prox_grad_upd_v0(w, v, thr, v_norm_deg)
            v_norm = torch.linalg.vector_norm(v_upd, dim=0, ord=v_norm_deg)  # [N,]
        elif isinstance(grouped_layer[0], nn.Conv2d) and isinstance(grouped_layer[1], nn.Conv2d):
            w_upd, v_upd = prox_grad_upd_v0_conv(w, v, thr, v_norm_deg)
            v_norm = torch.linalg.vector_norm(v_upd, dim=(0, 2, 3), ord=v_norm_deg)  # [N,]
        grouped_layer[0].weight.data = w_upd.data
        grouped_layer[1].weight.data = v_upd.data

        if mask is not None and mask_null is not None:
            new_mask = torch.where(v_norm <= 0, mask_null, mask)
            new_masks.append(new_mask)

    return model, new_masks


def prune_or_regularize_v2(mask, mask_null, model, thr, v_norm_deg):
    """This function prune the model by its magnitude， basically is what soft-thresholding does
    :param mask: the pervious mask
    :param model: the model to be pruned
    :param thr: if below the threshold, just set it to be zero
    :returns:
        model: the updated model (after apply the new mask)
        new_masks: the updated mask
    """

    new_masks = []
    for idx, layer in enumerate(model.scale_layers):
        s = layer.data
        s_upd = prox_grad_upd_v2(s, thr)
        layer.data = s_upd.data

        if mask is not None and mask_null is not None:
            new_mask = torch.where(s_upd <= 0, mask_null, mask)
            new_masks.append(new_mask)

    return model, new_masks
