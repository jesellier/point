
import numpy as np
import math
import numbers


def check_random_state_instance(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
    


#def getRandomSDMatrix(matrixSize, rng):
    #'random positive semi-define matrix (with inital A has rand[0,1] entry)
    #A = rng.rand(matrixSize, matrixSize)
    #B = np.dot(A, A.transpose())
    #return B

#def getSumProdMatrix(A,B):
    #s = 0
   # for i in range(len(A)):
        #for j in range(len(B)):
            #s = s + A[i,j]*B[i,j]
    #return s


#def ratioDerivative(kappa, n0, n1):
    #return the partial derivative of kappa[0]/kappa[1] for
    #der = 0
    #if n0 >= 2 :
        #return 0
    #der = ((-1)**n1) * math.factorial(n1) * ((kappa[1])**(-n1 - 1))
    #if n0 == 0 :
        #return der * kappa[0]

    #return der


    