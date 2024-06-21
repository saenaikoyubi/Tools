import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from SkewedT1d import Skewedt1d

class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs:torch.tensor):
        return -torch.sum(outputs)

class Gauss2d:
    def __init__(self):
        self.rho = np.nan
        self.delta = .0002
        self.span = 7

    def pdf(self, x, y):
        index = -1/(1-self.rho**2)*(x**2-2*self.rho*x*y+y**2)
        return 1/(2*np.pi) * 1/np.sqrt(1-self.rho**2) * np.exp(index)
    
    def fit(self, x, y):
        self.rho = np.corrcoef(x, y)[0,1]
        return True
    
    def mpdf(self, x):
        return (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)
    
    def mcdf(self, x):
        arrange = np.arange(-self.span, x, self.delta)
        return sum([self.mpdf(x=x_)*self.delta for x_ in arrange])
    
    def inverseMcdf(self, u):
        lower = -self.span if min(1-max(u),min(u)) < 1e-7 else -10
        arrange = np.arange(lower, -lower, self.delta)
        z = [self.mpdf(x=x_)*self.delta for x_ in arrange]
        mcdfList = np.cumsum(z)
        mcdfList = pd.Series(index=arrange, data=mcdfList)
        inverse = [list(mcdfList[mcdfList < u_].index)[-1] if len(list(mcdfList[mcdfList < u_].index)) != 0 else lower + self.delta/2 for u_ in u]
        return inverse

class Copula:
    def __init__(self, delta=.0005):
        self.gauss = Gauss2d()
        self.t1d_x = Skewedt1d()
        self.t1d_y = Skewedt1d()
        self.delta = delta
        self.span = 7

    def fit(self, x, y):
        print('fitting copula start')
        self.t1d_x.fit(x=x)
        t1d_x_lambda = self.t1d_x.lamb.to('cpu').detach().numpy().copy()[0]
        t1d_x_nu = self.t1d_x.nu.to('cpu').detach().numpy().copy()[0]
        print('t1d(x) param is lambda={:.4f} and nu={:.4f}'.format(t1d_x_lambda, t1d_x_nu))
        self.t1d_y.fit(x=y)
        t1d_y_lambda = self.t1d_y.lamb.to('cpu').detach().numpy().copy()[0]
        t1d_y_nu = self.t1d_y.nu.to('cpu').detach().numpy().copy()[0]
        print('t1d(y) param is lambda={:.4f} and nu={:.4f}'.format(t1d_y_lambda, t1d_y_nu))
        u = self.t1d_x.cdf(x)
        v = self.t1d_y.cdf(y)
        x = self.gauss.inverseMcdf(u=u)
        y = self.gauss.inverseMcdf(u=v)
        self.gauss.fit(x, y)
        print('gaussian copula param is rho={:.4f}'.format(self.gauss.rho))
        print('fitting copula end')
        print()
        return True
    
    def _mi_x(self, x, y):
        arrange1 = np.arange(-self.span, x, self.delta)
        z1 = sum([self.gauss.pdf(x=x_, y=y)*self.delta for x_ in arrange1])
        arrange2 = np.arange(x+self.delta, self.span, self.delta)
        z2 = sum([self.gauss.pdf(x=x_, y=y)*self.delta for x_ in arrange2])
        return z1 / (z1+z2) - 0.5
    
    def mi_x(self, x, y):
        return np.array([self._mi_x(x_, y_) for x_, y_ in zip(x, y)])

    def _mi_y(self, x, y):
        arrange1 = np.arange(-self.span, y, self.delta)
        z1 = sum([self.gauss.pdf(x=x, y=y_)*self.delta for y_ in arrange1])
        arrange2 = np.arange(y+self.delta, self.span, self.delta)
        z2 = sum([self.gauss.pdf(x=x, y=y_)*self.delta for y_ in arrange2])
        return z1 / (z1+z2) - 0.5
    
    def mi_y(self, x, y):
        return np.array([self._mi_y(x_, y_) for x_, y_ in zip(x, y)])
        


    
