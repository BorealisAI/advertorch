import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def pytorch_wrapper(func):
    def wrapped_func(x):
        x_numpy = x.cpu().data.numpy()
        output = func(x_numpy)
        output = torch.from_numpy(output)
        output = output.to(x.device)

        return output
    
    return wrapped_func

def norm(v):
    return torch.sqrt( (v ** 2).sum(-1) ) 

class GradientWrapper(torch.nn.Module):
    #facility for doing things in batch?
    def __init__(self, func):
        super().__init__()
        self.func = pytorch_wrapper(func)
        
        #Based on:
        #https://pytorch.org/docs/stable/notes/extending.html
        class _Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                #grad_est does not require grad
                output = self.func(input)
                grad_est = self.estimate_grad(input)
                ctx.save_for_backward(grad_est)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                #TODO: this is not general! May not work for images
                #Be careful about dimensions
                grad_est, = ctx.saved_tensors
                grad_input = None

                if ctx.needs_input_grad[0]:
                    grad_input = torch.bmm(grad_output.unsqueeze(1), grad_est)
                    grad_input = grad_input.squeeze(1)
                return grad_input

        self.diff_func = _Func.apply

    def batch_query(self, x):
        #TODO: accomodate images...
        n_batch, n_dim, nb_samples = x.shape
        x = x.permute(0, 2, 1).reshape(-1, n_dim)
        outputs = self.func(x) #shape [..., n_output]
        outputs = outputs.reshape(n_batch, nb_samples, -1)

        return outputs.permute(0, 2, 1)
    
    def estimate_grad(self, x):
        raise NotImplementedError

    def forward(self, x):
        #TODO: check compatibility with torch.no_grad()
        if not self.training:
            output = self.func(x)
        else:
            output = self.diff_func(x)

        return output

class FDWrapper(GradientWrapper):
    """
    Finite-Difference Estimator
    """
    def __init__(self, func, fd_eta=1e-3):
        super().__init__(func)
        self.fd_eta = fd_eta

    def estimate_grad(self, x):
        id_mat = torch.diag(torch.ones_like(x[0])) # shape [D,D]
        
        fxp = self.batch_query(
            x[:, :, None] + self.fd_eta * id_mat[None, :, :]
        )
        
        fxm = self.batch_query(
            x[:, :, None] - self.fd_eta * id_mat[None, :, :]
        )
        
        grad_est = (fxp - fxm) / (2.0 * self.fd_eta)
        return grad_est


class NESWrapper(GradientWrapper):
    def __init__(self, func, nb_samples, fd_eta=1e-3):
        super().__init__(func)
        self.nb_samples = nb_samples
        self.fd_eta = fd_eta

    def estimate_grad(self, x, prior=None):
        #TODO: adjust this so that it works with images...
        #x shape: [nbatch, ndim]
        ndim = np.prod(list(x.shape[1:]))

        # [nbatch, ndim, nsamples]
        exp_noise = x.new_full(tuple(x.shape) + (self.nb_samples,), 0)
        exp_noise.normal_()
        exp_noise /= (ndim ** 0.5)

        fxp = self.batch_query(
            x.unsqueeze(-1) + self.fd_eta * exp_noise
        )

        fxm = self.batch_query(
            x.unsqueeze(-1) - self.fd_eta * exp_noise
        )

        gx_s = (fxp - fxm) / (2.0 * self.fd_eta) #[nbatch, noutput, nsamples]
        
        grad_est = (gx_s[:, :, None, :] * exp_noise[:, None, :, :]).sum(-1)
        
        return grad_est






