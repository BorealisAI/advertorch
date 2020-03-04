#!/bin/bash

cd $1
git clone https://github.com/MadryLab/mnist_challenge
cd mnist_challenge
python fetch_model.py secret
