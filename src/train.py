# coding: utf-8
import argparse
import time
import math
import torch
import torch.optim as optim
import torch.nn as nn

import data
import model
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

import pickle

# transformer, rnn, lstm
model_slt= "transformer"

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--max_sql', type=int, default=256,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--emb_dim', type=int, default=256)
parser.add_argument('--use_pe', action="store_true")
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = True

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size,'valid':eval_batch_size}
data_loader = data.Corpus("../data/ptb", batch_size, args.max_sql)

########################################
# Build LMModel model (bulid your language model here)
if model_slt == "transformer":
    model = model.LMModel_transformer(nvoc = len(data_loader.vocabulary), num_layers = args.num_layers,
                      dim = args.emb_dim, nhead = args.num_heads)
elif model_slt == "rnn":
    model = model.LMModel_RNN(nvoc=len(data_loader.vocabulary), num_layers=args.num_layers,
                              dim=args.emb_dim)
elif model_slt == "lstm":
    model = model.LMModel_LSTM(nvoc=len(data_loader.vocabulary), num_layers=args.num_layers,
                               dim=args.emb_dim)
else:
    raise ValueError(f"Unknown model type: {model_slt}")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.CrossEntropyLoss()

# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def evaluate():
    data_loader.set_valid()
    data, target, end_flag = data_loader.get_batch()
    model.eval()
    idx = 0
    avg_loss = 0
    print(f"Validating")
    while not end_flag:
        with torch.no_grad():
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target = target.to(device)
            if model_slt != "transformer":
                decode,_ = model(data)
            else:
                decode = model(data)

            # Calculate cross-entropy loss
            loss = criterion(decode.view(decode.size(0) * decode.size(1), -1), target)
            avg_loss += loss
            idx += 1
    print(f"The average loss is {avg_loss / idx}")
    return math.exp(avg_loss.item() / idx)


# Train Function
def train():
    data_loader.set_train()
    data, target, end_flag = data_loader.get_batch()
    model.train()
    idx = 0
    avg_loss = 0
    while not end_flag:
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        target = target.to(device)
        if model_slt != "transformer":
            decode, _ = model(data)
        else:
            decode = model(data)

        # Calculate cross-entropy loss
        optimizer.zero_grad()
        loss = criterion(decode.view(decode.size(0)*decode.size(1), -1), target)
        loss.backward()
        optimizer.step()
        if (idx+1) % 50 == 0:
            print(f"The loss is {loss}")
        idx += 1
        avg_loss += loss
    return math.exp(avg_loss.item() / idx)


# Loop over epochs.
train_perplexity = []
valid_perplexity = []
for epoch in range(1, args.epochs+1):
    print(f"Start training epoch {epoch}")
    train_perplexity.append(train())
    valid_perplexity.append(evaluate())

print(f"Train Perpelexity {train_perplexity}")
print(f"Valid Perpelexity {valid_perplexity}")

def save_model(model, filename="model.pth", save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def plot_train_perpelexity():
    x = np.arange(1, args.epochs + 1)
    plt.plot(x, train_perplexity, label='Train Perplexity')
    plt.plot(x, valid_perplexity, label='Valid Perplexity')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.title('Train and Valid Perplexity')
    plt.legend()
    plt.show()

# 打印train.py 的参数
def print_args():
    print("args:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

print_args()
plot_train_perpelexity()
save_model(model, f"trained_lm_model_{model_slt}.pth",save_dir="../models")

with open('training_args.pkl', 'wb') as f:
    pickle.dump(args, f)
print("Training arguments saved to training_args.pkl")