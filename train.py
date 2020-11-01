from __future__ import print_function
import os
import sys
sys.path.append("..")
import argparse

import numpy as np
import pandas as pd
import csv

from sklearn.preprocessing import normalize
from sklearn.preprocessing import RobustScaler
import random

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from data import Dataset, getDataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Linear(nn.Module):
    
    def __init__(self, in_features, out_features, priors):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

        self.softplus = nn.Softplus()

    def reset_parameters(self):
        # posterior
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        self.bias_mu.data.normal_(*self.posterior_mu_initial)
        self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x, sample=True):
        
        # Weight sampling
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        bias_var = self.bias_sigma ** 2

        act_mu = F.linear(x, self.W_mu, self.bias_mu)
        act_var = 1e-16 + F.linear(x ** 2, self.W_sigma ** 2, bias_var)
        act_std = torch.sqrt(act_var)

        loss = self.kl_loss()

        if self.training or sample:
            # sampling된 weight로 값 구하기
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(device)
            return act_mu + self.softplus(act_std) * eps, loss
            # return act_mu + act_std * eps, loss
        else:
            return act_mu

    def calculate_kl(self, mu_q, sig_q, mu_p, sig_p):
        
        kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()

        return kl

    def kl_loss(self):
        
        kl = self.calculate_kl(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        kl += self.calculate_kl(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)

        return kl

class myModel(nn.Module):
    """
    data: [batch, 140]
    target: [batch, 3]
    """

    def __init__(self, in_dim, out_dim, priors):
        super(myModel, self).__init__()

        self.basis_function = nn.Linear(in_dim, 32)
        torch.nn.init.xavier_uniform_(
            self.basis_function.weight,
            gain=torch.nn.init.calculate_gain('tanh'))
        self.fc_a = Linear(32, 3, priors=priors)
        self.tanh = nn.Tanh()
        
    def forward(self, x):

        basis_output = self.tanh(self.basis_function(x))
        output, loss = self.fc_a(basis_output)

        return output, loss

class ELBO2(nn.Module):
    def __init__(self, train_size):
        super(ELBO2, self).__init__()
        self.train_size = train_size

    def forward(self, samples, target):
        
        assert not target.requires_grad
        
        mse_loss = 0.0
        # 이거는 N개를 샘플링해서 평균을 한다
        for sample in samples:
            mse_loss += torch.nn.MSELoss(reduction='mean')(sample, target) 
        mse_loss /= len(samples)

        return mse_loss

def getModel(in_dim, out_dim, priors):
    
    return myModel(in_dim, out_dim, priors)

def train_model(model, train_loader, criterion, optimizer):
    
    model.train()
    train_loss, mse_loss, kl_loss = 0.0, 0.0, 0.0
    acc_list, kl_list = list(), list()

    for i, (inputs, labels) in enumerate(train_loader, 1):

        inputs, labels = inputs.to(device), labels.to(device)
        output, kl_output = model(inputs)
        
        sample_list = [model(inputs) for _ in range(1)]
        output_list = [output for output, kl_output in sample_list]
        kl_output = sample_list[0][1]

        beta = 0.1 # 1/len(train_loader)
        kl = beta * kl_output
        mse = criterion(output_list, labels)        
        loss = mse + kl

        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data.numpy()
        mse_loss += mse
        kl_loss += kl

    return train_loss/len(train_loader), mse_loss/len(train_loader), kl_loss/len(train_loader), output_list[0], labels    

initial_lr = 1e-3
n_epochs = 500

train_dataset_path = 'dataset\\train_data.csv'
validation_dataset_path = 'dataset\\validation_data.csv'
test_dataset_path = 'dataset\\sample_evaluation_data.csv'
checkpoint_path = 'checkpoint' # file name

priors={
    'prior_mu': 0,
    'prior_sigma': 0.2,
    'posterior_mu_initial': (0, 0.2),  # (mean, std) normal_
    'posterior_rho_initial': (0, 0.2),  # (mean, std) normal_
}

model = getModel(in_dim=140, out_dim=3, priors=priors).to(device)
train_set = Dataset(train_dataset_path)
validation_set = Dataset(validation_dataset_path)
test_set = Dataset(test_dataset_path)

train_loader, validation_loader, test_loader = getDataLoader(train_set, validation_set, test_set)

criterion = ELBO2(len(train_set))
optimizer = Adam(model.parameters(), lr=initial_lr)
lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, verbose=True)

load_pretrained_model = False
if load_pretrained_model == True:
    state_dict = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(state_dict)

valid_loss_max = np.Inf
print("train start! -------------------------------------------------------------------")
for epoch in range(n_epochs):
    
    model.train()
    train_loss, train_mse, train_kl, train_output, train_labels = train_model(
        model, train_loader, criterion, optimizer)


    print('[training] Epoch: {} \tTotal: {:.2f} \tMSE: {:.2f} \tKL: {:.2f} \toutput: {:.2f} {:.2f} {:.2f} \tlabel: {:.2f} {:.2f} {:.2f}'.format(
        epoch, train_loss, train_mse, train_kl, 
        train_output[0][0], train_output[0][1], train_output[0][2], 
        train_labels[0][0], train_labels[0][1], train_labels[0][2]))
    
    #if epoch % 100 == 0:            
    model.eval()
    valid_loss, valid_mse, valid_kl, valid_output, valid_labels = train_model(
        model, validation_loader, criterion, optimizer)
    lr_sched.step(valid_loss)

    # print('\n\n[validation] \tTotal: {:.2f} \tMSE: {:.2f} \tKL: {:.2f} \toutput: {:.2f} {:.2f} {:.2f} \tlabel: {:.2f} {:.2f} {:.2f}'.format(
    #     valid_loss, valid_mse, valid_kl, 
    #     valid_output[0], valid_output[1], valid_output[2], 
    #     valid_labels[0], valid_labels[1], valid_labels[2]))

    # save model if validation accuracy has increased
    if valid_loss < valid_loss_max:
        print('Validation loss decreased ({:.2f} --> {:.2f}).  Saving model ...'.format(
            valid_loss_max, valid_loss))
        torch.save({'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}, 'checkpoint')
        valid_loss_max = valid_loss

print("train end! -------------------------------------------------------------------")