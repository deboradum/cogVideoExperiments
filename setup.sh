#!/bin/bash

sudo apt install vim -y

cd Pyramid-Flow
conda create -n pyramid python==3.8.10
conda activate pyramid

cd ../
pip install -r requirements.txt
