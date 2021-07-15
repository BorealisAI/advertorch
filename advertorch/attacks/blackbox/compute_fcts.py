import numpy as np
import torch as ch
from advertorch.utils import clamp as batch_clamp


def norm(t):
    """
    Return the norm of a tensor (or numpy) along all the dimensions except the first one
    """
    _shape = t.shape
    batch_size = _shape[0]
    num_dims = len(_shape[1:])
    if ch.is_tensor(t):
        norm_t = ch.sqrt(t.pow(2).sum(dim=[_ for _ in range(1, len(_shape))])).view([batch_size] + [1] * num_dims)
        norm_t += (norm_t == 0).float() * np.finfo(np.float64).eps
        return norm_t
    else:
        _norm = np.linalg.norm(t.reshape([batch_size, -1]), axis=1, keepdims=1).reshape([batch_size] + [1] * num_dims)
        return _norm + (_norm == 0) * np.finfo(np.float64).eps


def lp_step(x, g, lr, p):
    """
    performs lp step of x in the direction of g, where the norm is computed
    across all the dimensions except the first one (assuming it's the batch_size)

    Args:
        x: batch_size x dim x .. tensor (or numpy)
        g: batch_size x dim x .. tensor (or numpy)
        lr: learning rate (step size)
        p: 'inf' or '2'
    """
    if p == "inf":
        return linf_step(x, g, lr)
    elif p == "2":
        return l2_step(x, g, lr)
    else:
        raise Exception("Invalid p value")


def l2_step(x, g, lr):
    """
    performs l2 step of x in the direction of g, where the norm is computed
    across all the dimensions except the first one (assuming it's the batch_size)

    Args:
        x: batch_size x dim x .. tensor (or numpy)
        g: batch_size x dim x .. tensor (or numpy)
        lr: learning rate (step size)
    """
    if ch.is_tensor(x):
        x_ = x.clone()
    else:
        x_ = x.copy()
    return x_ + lr * g / norm(g)


def linf_step(x, g, lr):
    """
    performs linfinity step of x in the direction of g

    Args:
        x: batch_size x dim x .. tensor (or numpy)
        g: batch_size x dim x .. tensor (or numpy)
        lr: learning rate (step size)
    """
    if ch.is_tensor(x):
        x_ = x.clone()
        x_ = x_ + lr * ch.sign(g).to(x_.device)
    else:
        x_ = x.copy()
        x_ + lr * ch.sign(g)
    return x_


def l2_proj_maker(xs, eps):
    """
    makes an l2 projection function such that new points
    are projected within the eps l2-balls centered around xs
    """
    eps_ = eps.cpu().numpy()[0]
    if ch.is_tensor(xs):
        orig_xs = xs.clone()

        def proj(new_xs):
            delta = new_xs - orig_xs
            norm_delta = norm(delta)
            if np.isinf(eps_):  # unbounded projection
                return orig_xs + delta
            else:
                return (
                    orig_xs
                    + (norm_delta <= eps_).float() * delta
                    + (norm_delta > eps_).float() * eps_ * delta / norm_delta
                )

    else:
        orig_xs = xs.copy()

        def proj(new_xs):
            delta = new_xs - orig_xs
            norm_delta = norm(delta)
            if np.isinf(eps_):  # unbounded projection
                return orig_xs + delta
            else:
                return orig_xs + (norm_delta <= eps_) * delta + (norm_delta > eps_) * eps_ * delta / norm_delta

    return proj


def linf_proj_maker(xs, eps):
    """
    makes an linf projection function such that new points
    are projected within the eps linf-balls centered around xs
    """
    if ch.is_tensor(xs):
        orig_xs = xs.clone()

        def proj(new_xs):
            eps_ = eps.view(len(eps), 1).repeat(1, new_xs.shape[1])
            return orig_xs + batch_clamp(new_xs - orig_xs, -eps_, eps_)

    else:
        orig_xs = xs.copy()

        def proj(new_xs):
            return np.clip(new_xs, orig_xs - eps, orig_xs + eps)

    return proj


def sign(t, is_ns_sign=True):
    """
    Given a tensor t of `batch_size x dim` return the (non)standard sign of `t`
    based on the `is_ns_sign` flag

    Args:
        t: tensor of `batch_size x dim`
        is_ns_sign: if True uses the non-standard sign function
    """
    _sign_t = ch.sign(t) if ch.is_tensor(t) else np.sign(t)
    if is_ns_sign:
        _sign_t[_sign_t == 0.0] = 1.0
    return _sign_t
