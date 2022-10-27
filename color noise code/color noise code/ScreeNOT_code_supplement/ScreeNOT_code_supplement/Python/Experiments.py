"""
Helper code for generating experiments in paper

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
import numpy as np
from numpy.linalg import svd, norm
from numpy import sqrt
import matplotlib.pyplot as plt

def R0(y, fZ, gamma):
    x = Noise.invertSpike(y, fZ, gamma)
    return x**2

def R1(y, fZ, gamma):
    x = Noise.invertSpike(y, fZ, gamma)
    return x**2 + y**2 + 4*y * Opt.D(y, fZ, gamma)/Opt.Dd(y, fZ, gamma)

"""
Evaluates ASE[x|t] at the given thresholds ts.

Parameters:
    x: an array of signal singular values
    fZ: an empirical noise distribtution
    gamma: shape parameter
    ts: an array of thresholds
    
Returns:
    rs: an array such that rs[i] is ASE[x|t] evaluated at t=ts[i]
"""
def ASEt(x, fZ, gamma, ts):
    rs = np.zeros(ts.size)

    y = np.zeros(x.size)
    
    for i in range(0,x.size):
        y[i] = Noise.spikeForward(x[i], fZ, gamma)
        for j in range(0,ts.size):
            if y[i] > ts[j]:
                rs[j] = rs[j] + R1(y[i], fZ, gamma)
            else:
                rs[j] = rs[j] + R0(y[i], fZ, gamma)
    
    return rs
    
"""
Evaluates ASE^*[x].

Parameters:
    x: an array of signal singular values
    fZ: an empirical noise distribtution
    gamma: shape parameter    
"""
def ASEOpt(x, fZ, gamma):
    r = 0
    bulkEdge = np.max(fZ)
    tol = 0.01
    
    for i in range(0,x.size):
        y = Noise.spikeForward(x[i], fZ, gamma)
        if y < bulkEdge + tol:
            r = r + x**2
        else:
            r = r + np.min([R0(y,fZ,gamma), R1(y,fZ,gamma)])
    return r

"""
Evaluates SEn[x|t] at the given thresholds ts.

Parameters:
    X: signal matrix
    Y: measured matrix
    ts: an array of thresholds
    
Returns:
    rs: an array such that rs[i] is SEn[x|t] evaluated at t=ts[i]
"""
def SEnt(X, Y, ts):
    rs = np.zeros(ts.size)
    U, y, Vt = svd(Y)
    p = y.size
    pos = np.zeros(ts.size, dtype=int) # pos[i] is how many components we need to take to thershold at ts[i]
    SEk = (-1)*np.ones(y.size+1) # SEk[k] the MSE obtained by taking k empirical singular values.
                            # this array is filled on demand
    for i in range(0,ts.size):
        while y[pos[i]] > ts[i] and pos[i] < p:
            pos[i] = pos[i] + 1
    
    for i in range(0,ts.size):
        
        if SEk[pos[i]] < 0:
            Xest = U[:, :pos[i]] @ np.diag( y[:pos[i]] ) @ Vt[:pos[i], :]
            SEk[pos[i]] = norm(X-Xest)**2
        rs[i] = SEk[pos[i]]
        
    return rs
    

"""
Evaluates SEn^*[x].

Parameters:
    X: signal matrix
    r: the rank of X (required for performance purposes)
    Y: measured matrix
    
Returns:
    SE: the value of SEn^*[x]
    pos: how many singular values of Y are taken to achieve the optimum
"""
def SEnOpt(X, r, Y):
    Xenergy = norm(X)**2
    U, y, Vt = svd(Y)
    pos_bound = r+1
    while norm(y[(r+1):pos_bound])**2 <= Xenergy and pos_bound < y.size:
        # if condition doesn't happen, estimating Xhat=0 already has better MSE
        pos_bound = pos_bound + 1
    pos = -1
    SE = np.Inf
    for k in range(0,pos_bound):
        Xest = U[:, :k] @ np.diag( y[:k] ) @ Vt[:k, :]
        newSE = norm(X-Xest)**2
        if newSE <= SE:
            SE=newSE
            pos=k
    return SE, pos

#p=500
#gamma=1.0
#n=np.int(np.ceil(p/gamma))
#X=np.zeros((n,p))
#X[0,0]=3
#Z=Noise.noiseMarcenkoPastur(p,gamma)
#Y = X + Z
#_, fZ, _ = svd(Z)
#_, fY, _ = svd(Y)
#spacing = np.min( fY[:-1]-fY[1:] )
#bulkEdge=fZ[0]
#ts = np.arange(fY[20]-spacing, fY[0]+0.1, spacing/3)
#tsOracle = np.arange(bulkEdge+0.02, fY[0]+0.1, spacing/3)
#ASE = ASEt(np.array([3]), fZ, gamma, tsOracle)
#SEn = SEnt(X, Y, ts)
#plt.plot(ts, SEn)
#plt.plot(tsOracle, ASE)
#Aopt = ASEOpt(np.array([3]), fZ, gamma)
#Sopt, _ = SEnOpt(X, 1, Y)
##plt.axhline(y=Aopt,linestyle='-.')
##plt.axhline(y=Sopt,linestyle='--')
#plt.show()
