#!/bin/bash

cd $1
git clone https://github.com/MadryLab/cifar10_challenge
cd cifar10_challenge
python fetch_model.py secret
