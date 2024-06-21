import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Skewedt1d_(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.nu_ = nn.Parameter(torch.empty(1).uniform_(-1.1, -.1))
        self.lamb_ = nn.Parameter(torch.empty(1).uniform_(-.1, .1))
        self.nu = 2+torch.exp(self.nu_)
        self.lamb = torch.tanh(self.lamb_)
        self.device = device

    def gamma(self, x):
        return torch.exp(torch.lgamma(x))

    def forward(self, x):
        self.nu = 2+torch.exp(self.nu_)
        self.lamb = torch.tanh(self.lamb_)
        logc = torch.lgamma((self.nu+1)/2) - 1/2*torch.log(self.nu-2) -1/2*torch.log(torch.tensor(np.pi)) - torch.lgamma(self.nu/2)
        c = torch.exp(logc)
        a = 4*self.lamb*c*((self.nu-2)/(self.nu-1))
        b = torch.sqrt(1 + 3*self.lamb**2 - a**2)
        sign = torch.where(x < -a/b, 1., -1.)
        logt = torch.log(b) + logc - (self.nu+1)/2*torch.log(1+1/(self.nu-2)*((b*x+a)/(1-sign*self.lamb))**2)
        return logt
        
    def pdf(self, x):
        self.nu = 2+torch.exp(self.nu_)
        self.lamb = torch.tanh(self.lamb_)
        c = self.gamma((self.nu+1)/2) / (torch.sqrt(torch.pi*(self.nu-2)) * self.gamma(self.nu/2))
        a = 4*self.lamb*c*((self.nu-2)/(self.nu-1))
        b = torch.sqrt(1 + 3*self.lamb**2 - a**2)
        sign = torch.where(x < -a/b, 1., -1.)
        skewt = b*c*(1 + 1/(self.nu-2)*((b*x+a)/(1-sign*self.lamb))**2)**(-(self.nu+1)/2)
        return skewt
    
    def setParams(self, nu_, lamb_):
        self.nu_ = nu_
        self.lamb_ = lamb_
        self.nu = 2+torch.exp(self.nu_)
        self.lamb = torch.tanh(self.lamb_)
        return True
        
class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        return -torch.sum(outputs)

class Skewedt1d(nn.Module):
    def __init__(self, dx=.0005, device='cpu'):
        super().__init__()
        self.device = device
        self.skewt = Skewedt1d_(device=self.device).to(device=self.device)
        self.optimizer = torch.optim.Adam(self.skewt.parameters(), lr=1e-3)
        self.loss_fn = Loss()
        self.dx = dx

    def fit_step(self, x):
        self.skewt.train()
        outputs = self.skewt(x)
        self.optimizer.zero_grad()
        loss = self.loss_fn(outputs)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def fit(self, x):
        self.xmin = -np.max(np.abs(x))*1.5
        x = torch.tensor(x).to(device=self.device)
        epoch = 1
        min_loss = np.inf
        exit_counter = 0
        while True:
            loss = self.fit_step(x=x)
            if torch.any(torch.isnan(x)):
                print(x)
            if torch.isnan(loss):
                print('detect nan')
                return False
            if min_loss > loss:
                min_loss = loss
                exit_counter = 0
            if min_loss < loss:
                exit_counter += 1
                self.lamb_, self.nu_ = self.skewt.lamb_, self.skewt.nu_
                self.lamb, self.nu = self.skewt.lamb, self.skewt.nu
                
            if exit_counter >= 10:
                self.skewt.setParams(nu_=self.nu_, lamb_=self.lamb_)
                break
            epoch += 1
        return True
    
    def pdf(self, x):
        self.skewt.eval()
        x = torch.tensor(x).to(device=self.device)
        return self.skewt.pdf(x=x).to('cpu').detach().numpy().copy()
    
    def cdf(self, x):
        x_ = [np.arange(self.xmin, xstep, self.dx) for xstep in x]
        cdf = np.array([sum(self.pdf(x_list)*self.dx) for x_list in x_])
        return cdf
    
    def plot(self, x):
        x_ = np.arange(self.xmin, -self.xmin, self.dx)
        plt.title('pdf')
        y = self.pdf(x_)
        ymax = max(y)
        plt.ylim(0,ymax)
        plt.xlim(self.xmin, -self.xmin)
        plt.plot(x_, y)
        plt.hist(x, bins=100, density=True)
        plt.show()
        return True