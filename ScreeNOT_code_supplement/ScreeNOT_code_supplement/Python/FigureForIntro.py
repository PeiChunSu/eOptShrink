"""
Code for generating figures for example in intro

``ScreeNOT : Exact MSE-Optimal Singular Value Thresholding in Correlated Noise ''
by David L. Donoho, Matan Gavish, Elad Romanov. arxiv:2009.12297

Cite this as: 
Donoho, David L., Gavish, Matan and Romanov, Elad. (2020). Code Supplement for "ScreeNOT: Exact MSE-Optimal Singular Value Thresholding in Correlated Noise". Stanford Digital Repository. https://arxiv.org/abs/2009.12297

MIT License

Copyright (c) 2020 David L. Donoho, Matan Gavish, Elad Romanov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import OptimalAdaptiveSVThreshold as Opt
import Noise
import Experiments
import numpy as np
from numpy.linalg import svd, norm
from numpy import sqrt
import matplotlib.pyplot as plt


def AR1_Noise(n, p, rho):
    Z = np.zeros((n,p))
    for i in range(0,n):
        Z[i,0] = np.random.normal(0,1)
        for j in range(1,p):
            Z[i,j] = rho*Z[i,j-1] + (1-rho)*np.random.normal(0,1)
    return Z/sqrt(n)

np.random.seed(123456)


x  = np.array([1.0,2.0,3.0,4.0])
r = 4
k = 12

n = 500
p = 500
gamma = p/n
rho = 0.2
Z = AR1_Noise(n,p,rho)

G = np.random.normal(0,1,(n,p))
A, _, Bt = svd(G)
X = A[:,0:r] @ np.diag(x) @ Bt[0:r,:]
Y = X + Z

vals_count = 30 

_, fY, _ = svd(Y)

optLoss, optHowMany = Experiments.SEnOpt(X, r, Y)
_, Topt, _ = Opt.adaptiveHardThresholding(Y, k)



"""
IntroFig1
"""
plt.clf()
# plt.rc('text', usetex=True)
plt.plot(range(1,optHowMany+1), fY[0:optHowMany], linestyle='None', marker='o', color='red')
plt.plot(range(optHowMany+1,vals_count+1), fY[optHowMany:vals_count], linestyle='None', marker='o', color='blue')
plt.axhline(y=Topt, color='green', linestyle='--')
plt.xlabel(r'Component number')
plt.ylabel(r'Singular value')
plt.title(r'Scree plot: AR(1) noise')
plt.savefig('IntroFig1.pdf', bbox_inches='tight')


"""
IntroFig2
"""
plt.clf()
# plt.rc('text', usetex=True)
pseudoNoise = Opt.createPseudoNoise(fY, k)
edge = np.max(pseudoNoise)
thetas = np.arange(edge+0.05, Topt+0.3, 0.01)
Psis = np.zeros(thetas.shape)
for i in range(0,thetas.size):
    Psis[i] = Opt.F(thetas[i],pseudoNoise,gamma)
plt.plot(thetas,Psis)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\Psi(\theta)$')
plt.axhline(y=-4, color='blue', linestyle='--')
plt.axvline(x=Topt, color='green', linestyle='--')
plt.savefig('IntroFig2.pdf', bbox_inches='tight')


"""
IntroFig3
"""
plt.clf()
# plt.rc('text', usetex=True)
thetas = np.arange(fY[vals_count], fY[0]+0.5, 0.01)
SEs = Experiments.SEnt(X, Y, thetas)
plt.plot(thetas, SEs, label=r'$SE[\theta|X]$')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$SE[\theta|X]$')
plt.axvline(x=Topt, color='green', linestyle='--', label=r'$\hat{\theta}$')
plt.axhline(y=optLoss, color='red', linestyle='--', label=r'$\min_{\theta} \,\,SE[\theta|X]$')
plt.legend()
plt.savefig('IntroFig3.pdf', bbox_inches='tight')


np.random.seed()
