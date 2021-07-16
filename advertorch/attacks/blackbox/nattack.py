import numpy as np

import torch
import torch.nn as nn


from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin

from .utils import _check_param

npop = 300     # population size
sigma = 0.1    # noise standard deviation
alpha = 0.02  # learning rate
# alpha = 0.001  # learning rate
boxmin = 0
boxmax = 1
shift = (boxmin + boxmax) / 2. # 1/2 ... rescale based on clip_min, clip_max ... rename to shift
scale = (boxmax - boxmin) / 2. # 1/2 ... rename to scale

#scale = (clip_max - clip_min) / 2
#shift = (clip_max + clip_min) / 2

epsi = 0.031
epsilon = 1e-30 #numerical safety factor (buffer)

def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (np.log((1 + x) / (1 - x))) * 0.5

def encode_normal(z):
    return scale * np.tanh(z) + shift

def decode_input(x):
    return torch_arctanh((x - shift) / scale)

def l2_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        delta = new_x - orig
        out_of_bounds_mask = (norm(delta) > eps).float()
        x = (orig + eps*delta/norm(delta))*out_of_bounds_mask
        x += new_x*(1-out_of_bounds_mask)
        return x
    return proj

def linf_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        return orig + torch.clamp(new_x - orig, -eps, eps)
    return proj


#cifar images have 3 channels (RGB) and are 32 x 32

#linf and l2 projections

def loss_fn(outputs, y):
    npop, n_class = outputs.shape

    target_onehot = np.eye(n_class)[y].repeat(npop, 0)

    real = np.log((target_onehot * outputs).sum(1) + epsilon)
    other = np.log(((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0]+epsilon)

    loss1 = np.clip(real - other, 0.,1000)

    loss1 = loss_fn(outputs, y)

    return - 0.5 * loss1

class NAttack(Attack, LabelMixin):
    def __init__(
            self, predict, eps: float, order,
            loss_fn=None, 
            nb_samples=100,
            nb_iter=40,
            eps_iter=0.02,
            sigma=0.1,
            clip_min=0., clip_max=1.,
            targeted : bool = False
            ):

        super().__init__(predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.targeted = targeted
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.nb_samples = nb_samples
        self.order = order

        self.proj_maker = l2_proj if order == 'l2' else linf_proj

    def perturb(self, x, y):
        x, y = self._verify_and_process_inputs(x, y)
        x_adv = x.clone()

        eps = _check_param(self.eps, x.new_full((x.shape[0], )), 1, 'eps')
        clip_min = _check_param(self.clip_min, x, 'clip_min')
        clip_max = _check_param(self.clip_max, x, 'clip_max')

        proj_step = self.proj_maker(x, eps)

        shift = (clip_max + clip_min) / 2.
        scale = (clip_max - clip_min) / 2.

        #predict
        #mu_t = np.random.randn(ndim) * 0.001

        #[B, C, H, W]
        #mu_t = np.random.randn(1,3,32,32) * 0.001
        mu_t = np.random.randn(n_batch, n_dim) * 0.001

        for _ in range(n_iter):
            #[N, C, H, W]
            #Sample from N(0,I)
            gauss_samples = np.random.randn(npop, n_dim)# np.random.randn(npop, 3,32,32)

            #[N, C, H, W]
            #Compute gi = g(mu_t + sigma * samples)
            mu_samples = mu_t[None, :] + sigma * gauss_samples

            #[H, W, C] -> [C, H, W]
            #adv = torch_arctanh((x - shift) / scale)

            #x_decoded = decode_input(x)
            #print('adv', adv,flush=True)

            #[N, C, H, W]
            #inputimg = np.tanh(adv + mu_samples) * scale + shift
            #gi = encode_normal(x_decode + mu_samples)
            gi = encode_normal(mu_samples)

            #linf proj OR l2_proj
            #[N, C, H, W]
            #delta = gi - encode_normal(x_decode)
            #delta = gi - x

            x_adv = proj_step(gi)

            #batch_clamp?
            x_adv = torch.clamp(x_adv, clip_min, clip_max)


            #clipdist = np.clip(dist, -epsi, epsi)
            #clipinput = clipdist + encode_normal(x_decode)

            #clipinput = np.squeeze(clipinput)
            #clipinput = np.asarray(clipinput, dtype='float32')
            # input_var = autograd.Variable(torch.from_numpy(clipinput).cuda(), volatile=True)
            # #outputs = model(clipinput.transpose(0,2,3,1)).data.cpu().numpy()
            # outputs = model((input_var-means)/stds).data.cpu().numpy()
            # outputs = softmax(outputs)

            #shouldn't this go earlier?
            outputs = predict(x_adv)

            losses = loss_fn(outputs, y)
            #[N]
            z_score = (losses - np.mean(losses)) / (np.std(losses)+1e-7)

            #gauss_samples : [N, C, H, W]
            #z_score: N
            mu_t = mu_t + (alpha/(npop*sigma)) * (zscore @ gauss_samples)

        