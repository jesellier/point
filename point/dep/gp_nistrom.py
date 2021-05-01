# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:36:21 2021

@author: jesel
"""
import numpy as np
import scipy as scp

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from scipy.linalg import cholesky, cho_solve, solve_triangular

from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from point.utils import getRandomSDMatrix, getSumProdMatrix, ratioDerivative
from scipy.optimize import minimize #For optimizing
import time


def eigvalsh_to_eps(spectrum, cond=None, rcond=None):
        """
        Determine which eigenvalues are "small" given the spectrum.
        This is for compatibility across various linear algebra functions
        that should agree about whether or not a Hermitian matrix is numerically
        singular and what is its numerical matrix rank.
        This is designed to be compatible with scipy.linalg.pinvh.
        Parameters
        ----------
        spectrum : 1d ndarray
            Array of eigenvalues of a Hermitian matrix.
        cond, rcond : float, optional
            Cutoff for small eigenvalues.
            Singular values smaller than rcond * largest_eigenvalue are
            considered zero.
            If None or -1, suitable machine precision is used.
        Returns
        -------
        eps : float
            Magnitude cutoff for numerical negligibility.
        """
        if rcond is not None:
            cond = rcond
        if cond in [None, -1]:
            t = spectrum.dtype.char.lower()
            factor = {'f': 1E3, 'd': 1E6}
            cond = factor[t] * np.finfo(t).eps
        eps = cond * np.max(abs(spectrum))
        return eps
    
def pinv_1d(v, eps=1e-5):
    """
    A helper function for computing the pseudoinverse.
    Parameters
    ----------
    v : iterable of numbers
        This may be thought of as a vector of eigenvalues or singular values.
    eps : float
        Values with magnitude no greater than eps are considered negligible.
    Returns
    -------
    v_pinv : 1d float ndarray
        A vector of pseudo-inverted numbers.
    """
    return np.array([0 if abs(x) <= eps else 1/x for x in v], dtype=float)


    
class PSD(object):
    """
    Compute coordinated functions of a symmetric positive semidefinite matrix.
    This class addresses two issues.  Firstly it allows the pseudoinverse,
    the logarithm of the pseudo-determinant, and the rank of the matrix
    to be computed using one call to eigh instead of three.
    Secondly it allows these functions to be computed in a way
    that gives mutually compatible results.
    All of the functions are computed with a common understanding as to
    which of the eigenvalues are to be considered negligibly small.
    The functions are designed to coordinate with scipy.linalg.pinvh()
    but not necessarily with np.linalg.det() or with np.linalg.matrix_rank().
    Parameters
    ----------
    M : array_like
        Symmetric positive semidefinite matrix (2-D).
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are
        considered zero.
        If None or -1, suitable machine precision is used.
    lower : bool, optional
        Whether the pertinent array data is taken from the lower
        or upper triangle of M. (Default: lower)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite
        numbers. Disabling may give a performance gain, but may result
        in problems (crashes, non-termination) if the inputs do contain
        infinities or NaNs.
    allow_singular : bool, optional
        Whether to allow a singular matrix.  (Default: True)
    Notes
    -----
    The arguments are similar to those of scipy.linalg.pinvh().
    """

    def __init__(self, M, cond=None, rcond=None, lower=True,
                 check_finite=True, allow_singular=True):
        # Compute the symmetric eigendecomposition.
        # Note that eigh takes care of array conversion, chkfinite,
        # and assertion that the matrix is square.
        s, u = scp.linalg.eigh(M, lower=lower, check_finite=check_finite)

        eps = eigvalsh_to_eps(s, cond, rcond)
        if np.min(s) < -eps:
            raise ValueError('the input matrix must be positive semidefinite')
        d = s[s > eps]
        if len(d) < len(s) and not allow_singular:
            raise np.linalg.LinAlgError('singular matrix')
        s_pinv = pinv_1d(s, eps)
        U = np.multiply(u, np.sqrt(s_pinv))

        # Initialize the eagerly precomputed attributes.
        self.U = U
        # Initialize an attribute to be lazily computed.
        self._pinv = None

    @property
    def pinv(self):
        if self._pinv is None:
            self._pinv = np.dot(self.U, self.U.T)
        return self._pinv




class NystromKernel():
    
    from enum import Enum
    class SamplingMode(Enum):
        SAMPLING = 1
        GRID = 2
        PRE_ASSIGNED = 3
    
    def __init__(self, kernel, random_state = np.random.RandomState(), 
                 bounds = np.array([0,1]), noise = 0, lower=True,
                 check_finite=True, allow_singular= False):
        
        super().__init__()
        
        self.sample = None
        self.kernel = kernel
        self.latent = None
                 
        self._bounds = bounds
        self._random_state = random_state
        self._card = bounds[1] - bounds[0]

        self._n = None
        self._s = None
        self._U = None
        self._v = None
        self._vl = None
        
        self._noise = noise
        self._lower = lower
        self._check_finite = check_finite
        self._allow_singular= allow_singular
        
                 
    def fit(self, mode = SamplingMode.SAMPLING, n = 100, sample = None):
        if mode == NystromKernel.SamplingMode.SAMPLING :
            self._sample(n)
        elif mode == NystromKernel.SamplingMode.GRID :
            self._grid(n)
        elif mode == NystromKernel.SamplingMode.PRE_ASSIGNED:
            if sample is None :
                raise ValueError("No sample passed exception thrown")
            self.sample = sample
        else :
            raise ValueError("Mode not recognized")
            
        self._evd()
        
        return self
    
    def simulate_latent(self):
        if self.sample is None :
            raise ValueError("Sample must be assigned")
        
        self.latent = self._random_state.normal(loc=0.0, scale=1.0, size= self._n)
        self._vl = self._v @ self.latent
            
            
           
    def _sample(self, n):
        sample = self._random_state.uniform(self._bounds[0],  self._bounds[1], n)
        sample.resize(n,1)
        self.sample = sample
        
    def _grid(self, n):
        sample = np.linspace(self._bounds[0], self._bounds[1], n)
        sample.resize(n,1)
        self.sample = sample
        
    def _evd(self):
        
        K = self.kernel(self.sample)
        if self._noise != 0 :
           K[np.diag_indices_from(K)] += self._noise
        
        self._U, self._s, V  = scp.linalg.svd(K)
        self._n = len(self._s)

        s = self._s
        eps = eigvalsh_to_eps(s, None, None)
        if np.min(s) < -eps:
            raise ValueError('the input matrix must be positive semidefinite')
        d = s[s > eps]
        if len(d) < len(s) and not self._allow_singular:
            raise np.linalg.LinAlgError('singular matrix')

        #self._v  = np.multiply(self._U, np.sqrt(pinv_1d(s, eps)))
        self._v  = np.multiply(self._U, np.sqrt(1/self._s))
      
        
    def inv(self):
        return self._v @ self._v.T
    
    
    def f(self, x):
        return self.kernel(x, self.sample) @ self._vl
    

            

        
        
    
    
    
    
    
#%%%%%%%%%%%%%%%%%%%
rng  = np.random.RandomState(10)

kernel = 4 * RBF(length_scale=[5]) #+ RBF(length_scale=[0.2])
noise = 1e-5
#noise = 0

gp = GaussianProcessRegressor(
            kernel= kernel,
            alpha= noise,
            normalize_y= False,
            optimizer = None,
            random_state= rng
            )

space = (-10,10)
measure = space[1] - space[0]

N = 5000
X = rng.uniform(-1,  1, N)
X.resize(N,1)
Kxx = kernel(X)

m = 500
n = NystromKernel(kernel, noise = noise, allow_singular= False)
n.fit(n = m)
n.simulate_latent()


x = np.array([1.2])
Knn, grad = n.kernel(n.sample, eval_gradient= True)
Kxn = n.kernel(x, n.sample)



# SECOND TERM
#final = np.zeros(n._n)
#now = time.time() 
#for i in range(500):
    #k = 0
    #tmp = grad[:,:,k] @ n._U[:,i]
    #term1 =  (n._U[:,i].T @ tmp) * n._U[:,i]
    #pinv = np.linalg.pinv(Knn - n._s[i] * np.eye(n._n))
    #term3 = n.latent[i] /(2* np.square(n._s[i])) * ( (1/n._s[i]) * term1 - pinv @ tmp)
    #final[i] = Kxn @ term3
#print(time.time()  - now)
    
### 1ER TERM
now = time.time() 
v = np.multiply(n._U,n.latent)

sum2 = ((v @ v.T) * grad[:,:,0].T).sum()
print(time.time()  - now)




#optimMin = minimize(n.f, x0 = [0.0], bounds= [space]);
#optimMax = minimize(lambda x : - n.f(x), x0 = [0.0], bounds= [space]);
#lamMax = 30 * max(np.abs(optimMax.fun), np.abs(optimMin.fun))**2; #retrieve minimum value found by minimize

#test
#fx = n.f(X)
#max_out = max(fx)
#min_out = min(fx)
#lamEst=  max(np.abs(max_out), np.abs(min_out))**2

#print("sampling.max : ", - optimMax.fun)
#print("quantile.up : ", -np.quantile(-fx, 0.001))
#print("sampling.min : ",  optimMin.fun)
#print("quantile.down : ", np.quantile(fx, 0.001))


#numPoints = rng.poisson(size= 1, lam = lamMax * measure)[0]



#print("num.Points : ", numPoints)
#Xpp = rng.uniform(space[0], space[1], size=(numPoints, 1))
#lamb =  n.f(Xpp)**2
#tmp = rng.uniform(0,1, numPoints)< lamb.reshape(numPoints)/lamMax; 
#x_thinned = Xpp[~tmp]
#x_points = Xpp[tmp]

#plt.plot(x_points, [0] * len(x_points), 'k|', ms = 12, mew = 2.5)
#plt.axis([-10, 10, -2, 2])
#plt.axhline(0, color='black', linewidth = 0.5)
#plt.show()


#Knn = n.kernel(n.sample)
#Kxn = kernel(X,n.sample)
#Kest = Kxn @ n.inv() @ Kxn.T

#A = kernel(n.sample)
#A[np.diag_indices_from(A)] += noise
#psd = PSD(A, allow_singular=False)

#U,s,V = scp.linalg.svd(A)  
#Ainv1 = np.linalg.inv(A)
#Ainv2 = U @ np.diag(1/s) @ np.transpose(U)
#Ainv3 = psd.pinv

#test = A @ n.inv()
#test1 = A @ Ainv1
#test2 = A @ Ainv2
#test3 = A @ Ainv3







#%%%%%%%%%%%%%%%%%%%  1 dim Plot
#N = 30
#X = np.linspace(-3, 3, N)
#X1g, X2g = np.meshgrid(X, X)
#X.resize(N,1)
#Z, grad = kernel(X, eval_gradient=True)

# Create a surface plot and projected filled contour plot under it.
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.contour3D(X1g, X2g , Z, 50, cmap='binary')
#ax.plot_surface(X1g, X2g, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
#ax.view_init(60, 35)
#plt.show()
#%%%%%%%%%%%%%%%%%%%  2 dim Plot
#N = 10
#space = np.linspace(-3, 3, N)
#X1g, X2g = np.meshgrid(space, space)
#X = np.array([X1g.flatten(), X2g.flatten()]).T
#Z, grad = kernel(X, eval_gradient=True)

# Create a surface plot and projected filled contour plot under it.
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax = plt.axes(projection='3d')
#ax.contour3D(X1g, X2g , Z, 50, cmap='binary')
#space2 = np.linspace(0, N**2, N**2)
#m1, m2 = np.meshgrid(space2, space2)
#ax.plot_surface(m1, m2, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
#ax.view_init(65, 45)
#plt.show()