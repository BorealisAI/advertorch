# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys

import torch
import torchvision

import advertorch
from advertorch.attacks.utils import multiple_mini_batch_attack


def get_benchmark_sys_info():
    rval = "#\n#\n"
    rval += ("# Automatically generated benchmark report "
             "(screen print of running this file)\n#\n")
    uname = os.uname()
    rval += "# sysname: {}\n".format(uname.sysname)
    rval += "# release: {}\n".format(uname.release)
    rval += "# version: {}\n".format(uname.version)
    rval += "# machine: {}\n".format(uname.machine)
    rval += "# python: {}.{}.{}\n".format(
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro)
    rval += "# torch: {}\n".format(torch.__version__)
    rval += "# torchvision: {}\n".format(torchvision.__version__)
    rval += "# advertorch: {}\n".format(advertorch.__version__)
    return rval


def _calculate_benchmark_results(
        model, loader, attack_class, attack_kwargs, norm, device, num_batch):
    adversary = attack_class(model, **attack_kwargs)
    label, pred, advpred, dist = multiple_mini_batch_attack(
        adversary, loader, device=device, norm=norm, num_batch=num_batch)
    accuracy = 100. * (label == pred).sum().item() / len(label)
    attack_success_rate = 100. * (label != advpred).sum().item() / len(label)
    dist = None if dist is None else dist[(label != advpred) & (label == pred)]
    return len(label), accuracy, attack_success_rate, dist


def _generate_basic_benchmark_str(
        model, loader, attack_class, attack_kwargs, num, accuracy,
        attack_success_rate):
    rval = ""
    rval += "# attack type: {}\n".format(attack_class.__name__)

    prefix = " attack kwargs: "
    count = 0
    for key in attack_kwargs:
        this_prefix = prefix if count == 0 else " " * len(prefix)
        count += 1
        rval += "#{}{}={}\n".format(this_prefix, key, attack_kwargs[key])

    rval += "# data: {}, {} samples\n".format(loader.name, num)
    rval += "# model: {}\n".format(model.name)
    rval += "# accuracy: {}%\n".format(accuracy)
    rval += "# attack success rate: {}%\n".format(attack_success_rate)
    return rval


def benchmark_attack_success_rate(
        model, loader, attack_class, attack_kwargs,
        device="cuda", num_batch=None):
    num, accuracy, attack_success_rate, _ = _calculate_benchmark_results(
        model, loader, attack_class, attack_kwargs, None, device, num_batch)
    rval = _generate_basic_benchmark_str(
        model, loader, attack_class, attack_kwargs, num, accuracy,
        attack_success_rate)
    return rval


def benchmark_margin(
        model, loader, attack_class, attack_kwargs, norm,
        device="cuda", num_batch=None):

    num, accuracy, attack_success_rate, dist = _calculate_benchmark_results(
        model, loader, attack_class, attack_kwargs, norm, device, num_batch)
    rval = _generate_basic_benchmark_str(
        model, loader, attack_class, attack_kwargs, num, accuracy,
        attack_success_rate)

    rval += "# Among successful attacks ({} norm) ".format(norm) + \
        "on correctly classified examples:\n"
    rval += "#    minimum distance: {:.4}\n".format(dist.min().item())
    rval += "#    median distance: {:.4}\n".format(dist.median().item())
    rval += "#    maximum distance: {:.4}\n".format(dist.max().item())
    rval += "#    average distance: {:.4}\n".format(dist.mean().item())
    rval += "#    distance standard deviation: {:.4}\n".format(
        dist.std().item())

    return rval
