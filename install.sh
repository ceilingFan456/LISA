#!/bin/bash

pip3 install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

CUDA_VISIBLE_DEVICES=0 python chat.py --version='xinlai/LISA-13B-llama2-v1'

CUDA_VISIBLE_DEVICES=0 python inference.py --version='xinlai/LISA-13B-llama2-v1'

## around 4 hours. 
