import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from matplotlib import pyplot
import torch.nn.functional as f
import random
from datetime import datetime
random.seed(datetime.now().timestamp())
import torch.nn as nn

seed = 6565

class neural_net(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(neural_net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(x))   # → shape (hidden_dim,)
        y = self.fc2(h)               # → shape (1,)
        return y.squeeze()            # → shape ()

class mdp:
    def __init__(self, S, gamma):
      self.S = S
      self.gamma = gamma
      self.r = []
      self.P = []
      self.d = []
      self.Vparam = []
      self.stat_dist = []

    def generate_P(self, rnd = True):
        if rnd:
          torch.manual_seed(seed)
          self.P = torch.rand((self.S, self.S,))
          self.P = f.normalize(self.P, 1, 1).type(torch.FloatTensor)
        else:
          z = 6
          self.P = torch.zeros((self.S, self.S))
          for i in range(self.S):
            j = random.sample(range(self.S), z)
            self.P[i,j] = 1.0/z
          self.P = f.normalize(self.P, 1, 1).type(torch.FloatTensor)

    def generate_r(self, rnd = True):
        if rnd:
          torch.manual_seed(seed)
          self.r = torch.rand(self.S)
        else:
          phi = random_feature(self, self.d)
          I = torch.eye(self.S).type(torch.FloatTensor)
          self.Vparam = 3.0*torch.rand((self.d, 1))/np.sqrt(self.d)
          # self.Vparam = 4.0*(torch.rand((self.d, 1)) < 0.5)
          self.r = 32*(I-self.gamma*self.P)@phi@self.Vparam

    def exact_val_func(self):
      I = torch.eye(self.S).type(torch.FloatTensor)
      self.val_func = torch.inverse(I-self.gamma*self.P).type(torch.FloatTensor)@self.r
    
    def compute_stat_dist(self):
      _, Q = torch.linalg.eig(torch.transpose(self.P, 0, 1))
      self.stat_dist = torch.real(Q[:, 0]/torch.sum(Q[:,0]))

def random_feature(mdp0, d, crafted=False):
  torch.manual_seed(seed)
  phi = torch.randn(mdp0.S, d)
  phi = f.normalize(phi, 2, 1)
  return phi

def isfullrank(mdp0, d):
  U = torch.zeros((d, d))
  for s in range(mdp0.S):
    U = U + mdp0.stat_dist[s]*torch.reshape(phi[s,:],(d, 1))@torch.reshape(phi[s,:],(1, d))
  L, V = torch.linalg.eig(U)
  print(L)

def preconditioner(mdp0, d):
    U = torch.zeros((d, d))
    for s in range(mdp0.S):
        U = U + mdp0.stat_dist[s]*torch.reshape(phi[s,:],(d, 1))@torch.reshape(phi[s,:],(1, d))
    return torch.linalg.inv(U)

def bellman_mse(mdp0, est_value_func):
  S = mdp0.S
  diff = torch.reshape(mdp0.val_func,(S, 1))-torch.reshape(est_value_func,(S, 1))
  mse = torch.reshape(diff,(1, S))@torch.diag(mdp0.stat_dist)@torch.reshape(diff,(S,1))
  return mse

def bellman_mse_neural(mdp0, model):
  S = mdp0.S
  est_value_func = torch.zeros(S)
  phi = random_feature(mdp0, mdp0.d)
  for s in range(S):
     est_value_func[s] = f_new = model(phi[s,:]).detach().item()
  diff = torch.reshape(mdp0.val_func,(S, 1))-torch.reshape(est_value_func,(S, 1))
  mse = torch.reshape(diff,(1, S))@torch.diag(mdp0.stat_dist)@torch.reshape(diff,(S,1))
  return mse

d = 16

mdp0 = mdp(1024, 0.9)
mdp0.d = d
mdp0.generate_P(rnd=True)
mdp0.generate_r(rnd=False)
mdp0.compute_stat_dist()
mdp0.exact_val_func()

phi = random_feature(mdp0, mdp0.d)

m = 12
model = neural_net(d, m)
dummy_model = neural_net(d, m)
T = 1000
run_error = torch.zeros(T, 1)

rho = 30.0
lr = 0.01

theta0 = 2.0*torch.ones(d, 1)
def run_td_learning(mdp0, T):
  theta = theta0
  avg_theta = torch.zeros(d, 1)  
  theta_prev = theta
  run_error = torch.zeros(T, 1)
  grad_norm = torch.zeros(T, 1)
  theta1 = torch.zeros_like(model.fc1.weight.data)
  theta2 = torch.zeros_like(model.fc2.weight.data)
  s_cur = 0
  s_new = np.random.choice(np.arange(0, mdp0.S), p=mdp0.P[s_cur, :].numpy())
  for t in range(T):
    model.zero_grad()
    theta_prev = theta
    f_cur = model(phi[s_cur,:])
    f_new = model(phi[s_new,:]).detach().item()
    f_cur.backward()
    temp_diff = mdp0.r[s_cur]+mdp0.gamma*f_new-f_cur
    model.fc1.weight.data = model.fc1.weight.data + lr * temp_diff*model.fc1.weight.grad
    model.fc2.weight.data = model.fc2.weight.data + lr * temp_diff*model.fc2.weight.grad
    s_cur = s_new
    theta1 = (t*theta1 + model.fc1.weight.data.detach())/(t+1.0)
    theta2 = (t*theta2 + model.fc2.weight.data.detach())/(t+1.0)
    dummy_model.fc1.weight.data = theta1
    dummy_model.fc2.weight.data = theta2
    s_new = np.random.choice(np.arange(0, mdp0.S), p=mdp0.P[s_cur, :].numpy())
    run_error[t] = bellman_mse_neural(mdp0, dummy_model)
    print(t, run_error[t])
  return run_error

times = 1000
seed_sim = 1212

run_error = torch.zeros(T, 1)

new = run_td_learning(mdp0, T)

run_error = run_error+new

plt.figure(0)
plt.plot(range(T), run_error/times, 'r-', marker='o',markevery=2500)

plt.show()
