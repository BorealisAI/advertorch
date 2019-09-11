#!/bin/bash

cd $1
git clone git@github.com:MadryLab/cifar10_challenge.git
cd cifar10_challenge
python fetch_model.py secret
