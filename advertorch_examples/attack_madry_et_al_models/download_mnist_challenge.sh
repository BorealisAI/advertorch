#!/bin/bash

cd $1
git clone git@github.com:MadryLab/mnist_challenge.git
cd mnist_challenge
python fetch_model.py secret
