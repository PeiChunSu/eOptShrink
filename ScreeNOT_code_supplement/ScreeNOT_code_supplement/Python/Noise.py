"""
Helper code for generating noise matrices for the experiments

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

import numpy as np
from numpy import sqrt
from numpy.linalg import svd
import matplotlib.pyplot as plt

from OptimalAdaptiveSVThreshold import D, Dd

    
"""
Samples a (Gaussian) random matrix with correlated rows.

Parameters:
    gamma: shape parameter, so that p=fT.size and n=p/gamma
    fT: singular values of the matrix T
"""
def sampleCorrelatedNoiseMatrix(gamma, fT):
    p = fT.size
    n = np.int(np.ceil(p/gamma))
    Z = np.random.normal(0, 1/sqrt(n), (n,p))
    return np.diag( sqrt(fT) ) @ Z


"""
Esimates the inverse spike-forward mapping: y_{i,n} ->> x_i
WARNING: should not work well for spikes close to the bulk edge. this is because 
    we are using an empirical distribution, for which the D-transform always has 
    a (y-bulkEdge)^{-1} singularity near the edge. In particular, this does not compute
    the BBP location reliably.

Parameters:
    y: an observe spike to invert. assumes y > bulkEdge(fZ)
    fZ: an empirical noise distribution
    gamma: shape parameter
"""
def invertSpike(y, fZ, gamma):
    bulkEdge = np.max(fZ)
    assert( y > bulkEdge )
    return 1/np.sqrt(D(y,fZ,gamma))

"""
Evalutes the spike-forward mapping x -> Y(x), using the empirical distribution fZ:
    1/x^2 = D_gamma(y;fZ)
WARNING: this is highly inaccurate for x close to the bulk edge.
"""
def spikeForward(x, fZ, gamma):
    assert(x>0)
    bulkEdge = np.max(fZ)
    low = bulkEdge
    high = 2*bulkEdge
    while D(high, fZ, gamma) > 1/x**2:
        low = high
        high = 2*high
    eps = 1e-6
    while high-low > eps:
        mid = (high+low)/2
        if D(mid, fZ, gamma) > 1/x**2:
            low=mid
        else:
            high=mid
    return mid
    

"""
Estimate the location of the BBP threshold, using the inverse spike mapping evaluated 
close to the bulk edge. This is not accurate at all! see comment on function invertSpike.

Parameters:
    fZ: an empirical noise distribution
    gamma: shape parameter
"""
def estimateBBP(fZ, gamma):
    fZ = np.sort(fZ)
    diff = fZ[-1] - fZ[-2]
    steps = 2
    # take the inverse spike at bulkEdge + steps*diff, that is, steps time the difference
    #   between the two top singular values
    xBBP = invertSpike(fZ[-1] + steps*diff, fZ, gamma)
    return xBBP

"""
Code to generate specific noise matrices is here.
Each of these functions get parameters p and gamma, and produces and samples a noise matrix
according to the appropriate distribution.
"""

# Marcenko-Pastur law
def noiseMarcenkoPastur(p, gamma):
    fT = np.ones(p)
    return sampleCorrelatedNoiseMatrix(gamma, fT)
noiseMarcenkoPastur.name = 'Marcenko-Pastur'

# Correlated rows, with dF^T an equal mixture of two atoms: \delta_1 and \delta_10
def noiseMix2(p, gamma):
    fT = np.ones(p)
    fT[0:int(p/2)]=10.0
    return sampleCorrelatedNoiseMatrix(gamma, fT)
noiseMix2.name = 'Mix2'

# Correlated rows, with dF^T the uniform distribution on [1,10]
def noiseUnif_1_to_10(p, gamma):
    fT = np.ones(p)
    for l in range(0,p):
        fT[l] = fT[l] + 9*l/p
    return sampleCorrelatedNoiseMatrix(gamma, fT)
noiseUnif_1_to_10.name = 'Unif[1,10]'

# Noise matrix is an identity matrix (padded with zero)
def noisePaddedIdentity(p, gamma):
    n = np.int(np.ceil(p/gamma))
    assert( gamma <= 1)
    Z = np.vstack( [np.eye(p), np.zeros((n-p,p))] )
    return Z
noisePaddedIdentity.name = 'PaddedIdentity'

# Correlated rows, with dF^T the law of a chi-squared random variable with
# 10 degrees of freedom (normalized to have variance 1)
def noiseChi10(p, gamma):
    fT = np.random.normal(0, 1, (p,10))
    fT = np.mean(fT**2, axis=1)
    return sampleCorrelatedNoiseMatrix(gamma, fT)
noiseChi10.name = 'Chi10'

# Fisher matrix with parameter beta=3. That is, Z=WS^{-1/2} where S is a MP law
# with shape beta=3
def noiseFisher3n(p, gamma):
    T = np.random.normal(0, 1/sqrt(3*p), (3*p, p) )
    _, fT, _ = svd(T)
    fT = 1.0/fT
    return sampleCorrelatedNoiseMatrix(gamma, fT)
noiseFisher3n.name = 'Fisher3n'



