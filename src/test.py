# @Time     : 2025/5/7 22:53
# @Author   : Kun Lin
# @Filename : test.py.py


import torch
print(torch.__version__)

if torch.cuda.is_available():
    print('CUDA works')