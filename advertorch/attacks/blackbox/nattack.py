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

    target_onehot = torch.eye(n_class)[y]#.repeat(npop, 0)

    real = np.log((target_onehot * outputs).sum(1) + epsilon)
    other = np.log(((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0]+epsilon)

    loss1 = np.clip(real - other, 0.,1000)

    #loss1 = loss_fn(outputs, y)

    #TODO: sign based on target or untarget
    return -0.5 * loss1

class NAttack(Attack, LabelMixin):
    def __init__(
            self, predict, eps: float, order='linf',
            loss_fn=None, 
            nb_samples=100,
            nb_iter=40,
            eps_iter=0.02,
            sigma=0.1,
            clip_min=0., clip_max=1.,
            targeted : bool = False,
            npop = 300
            ):

        super().__init__(predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.targeted = targeted
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.nb_samples = nb_samples
        self.order = order
        self.npop = npop

        self.proj_maker = l2_proj if order == 'l2' else linf_proj

    def perturb(self, x, y):
        #[B, F]
        x, y = self._verify_and_process_inputs(x, y)
        x_adv = x.clone()

        n_batch, n_dim = x.shape

        eps = _check_param(self.eps, x.new_full((x.shape[0],) , 1), 'eps')
        #[B, F]
        clip_min = _check_param(self.clip_min, x, 'clip_min')
        clip_max = _check_param(self.clip_max, x, 'clip_max')

        proj_step = self.proj_maker(x, eps)

        #[B,F]
        shift = (clip_max + clip_min) / 2.
        scale = (clip_max - clip_min) / 2.

        x_scaled = torch_arctanh((x - shift) / scale)
        #TODO: shouldn't this be equal to x_scaled? Something to test
        x_unscaled = np.tanh(x_scaled) * scale + shift

        y_repeat = y.repeat(self.npop, 1).T.flatten()

        #Fleet Foxes - the shrine/the argument

        #predict
        #mu_t = np.random.randn(ndim) * 0.001

        #[B,F]
        mu_t = np.random.randn(n_batch, n_dim) * 0.001

        for _ in range(self.nb_iter):
            #Sample from N(0,I)
            #[B, N, F]
            gauss_samples = np.random.randn(n_batch, self.npop, n_dim)

            #Compute gi = g(mu_t + sigma * samples)
            #[B, N, F]
            mu_samples = mu_t[:, None, :] + sigma * gauss_samples

            #[B, N, F]
            adv = np.tanh(x_scaled[:, None, :] + mu_samples) * scale[:, None, :] + shift[:, None, :]


            #projection step
            #[B, N, F]
            dist = adv - x_unscaled[:, None, :]
            #[B, N, F]
            clipdist = np.clip(dist, -epsi, epsi)
            #[B, N, F]
            adv = (clipdist + x_unscaled[:, None, :])

            #x_decoded = decode_input(x)
            #print('adv', adv,flush=True)

            #[N, C, H, W]
            #inputimg = np.tanh(adv + mu_samples) * scale + shift
            #gi = encode_normal(x_decode + mu_samples)
            #gi = encode_normal(mu_samples)

            #linf proj OR l2_proj
            #[N, C, H, W]
            #delta = gi - encode_normal(x_decode)
            #delta = gi - x

            #x_adv = proj_step(gi)

            #batch_clamp?
            #x_adv = torch.clamp(x_adv, clip_min, clip_max)


            #clipdist = np.clip(dist, -epsi, epsi)
            #clipinput = clipdist + encode_normal(x_decode)

            #clipinput = np.squeeze(clipinput)
            #clipinput = np.asarray(clipinput, dtype='float32')
            # input_var = autograd.Variable(torch.from_numpy(clipinput).cuda(), volatile=True)
            # #outputs = model(clipinput.transpose(0,2,3,1)).data.cpu().numpy()
            # outputs = model((input_var-means)/stds).data.cpu().numpy()
            # outputs = softmax(outputs)

            #shouldn't this go earlier?
            adv = adv.reshape(-1, n_dim)
            outputs = self.predict(adv)
            #[B, N, C]
            #outputs = outputs.reshape(n_batch, self.npop, -1)
            #

            #[B * N]
            losses = loss_fn(outputs, y_repeat)
            #[B, N]
            losses = losses.reshape(n_batch, self.npop)
            
            #[N]
            #z_score = (losses - np.mean(losses)) / (np.std(losses) + 1e-7)
            #[B, N]
            z_score = (losses - losses.mean(1)[:, None]) / (losses.std(1)[:, None] + 1e-7)
            z_score = z_score.cpu().numpy()

            #mu_t: [B, F]
            #gauss_samples : [B,N,F]
            #z_score: [B,N]
            mu_t = mu_t + (alpha/(self.npop*sigma)) * (z_score[:, :, None] * gauss_samples).sum(1)

        #This isn't the adversarial example, you need to store adv. examples in the above....
        #these are triggered if misclassified
        adv = np.tanh(x_scaled + mu_t) * scale + shift
        return adv

        