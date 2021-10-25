import numpy as np
import math
from numpy.linalg import norm
from scipy.optimize import minimize

def labels(b_init, T):
    """
    Generate (T+1,n)-array of label vectors in {0,1}^n, with 0th row b_init and consecutive rows changing by one entry.  
    Inputs:
      - b_init: initial row vector in {0,1}^n
      - T: number of rows to generate after b_init
    """
    n = b_init.size
    b = [b_init]
    for t in range(1,T+1):
        x = b[t-1].copy()
        i = np.random.randint(0,n)
#         i = np.random.choice(range(0,n), size = int(np.floor(np.sqrt(n))), replace = False)
        x[i] = 1 - x[i]
        b.append(x)
    b = np.vstack(b)
    return b

def obj(t, x, A, b, mu):
    """
    Evaluate the t^th LR objective at x, given design matrix A, label sequence b, and regularization parameter mu
    Inputs:
      - t: time index, range 0 to T
      - x: d-dimensional input vector
      - A: (n,d) design matrix defining LR objectives
      - b: (T+1,n)-array of n-dimensional label vectors
      - mu: regularization parameter
    """
    n = A.shape[0]
    Ax = np.dot(A, x)
    return (1/n)*(np.sum(np.log(1 + np.exp(Ax))) - np.dot(b[t], Ax)) + (mu/2)*(norm(x)**2)
        
def grad(t, x, A, b, mu): 
    """
    Evaluate the gradient of the t^th LR objective at x, given design matrix A and label sequence b
    Inputs:
      - t: time index, range 0 to T
      - x: d-dimensional input vector
      - A: (n,d) design matrix defining LR objectives
      - b: (T+1,n)-array of n-dimensional label vectors
      - mu: regularization parameter
    """
    n = A.shape[0]
    Ax = np.dot(A, x)
    return (1/n)*np.dot(A.T, np.divide(np.exp(Ax), 1 + np.exp(Ax)) - b[t]) + mu*x

def samplegrad(t, x, A, b, mu): 
    """
    Evaluate an approximate gradient of the t^th LR objective at x, given design matrix A and label sequence b, using one of the objective summands chosen uniformly at random. 
    Inputs:
      - t: time index, range 0 to T
      - x: d-dimensional input vector
      - A: (n,d) design matrix defining LR objectives
      - b: (T+1,n)-array of n-dimensional label vectors
      - mu: regularization parameter
    """
    n = A.shape[0]
    i = np.random.randint(0,n)
    a_i = A[i,:]
    return np.divide(1, 1 + np.exp(-np.dot(a_i,x)))*a_i - b[t,i]*a_i + mu*x

def mins(A, b, mu):
    """
    Compute (T+1,d)-array of d-dimensional minimizers of LR objectives defined by A, b, and mu
    Inputs:
      - A: (n,d) design matrix defining LR objectives
      - b: (T+1,n)-array of n-dimensional label vectors
      - mu: regularization parameter
    Output:
      - (T+1,d)-array with t^th row the minimizer of the t^th LR objective defined by A, b, and m
    """
    T = b.shape[0] - 1
    d = A.shape[1]
    x0 = np.zeros(d)
    mins = []
    for t in range(0,T+1):
        def fun_t(x):
            return obj(t, x, A, b, mu)
        def jac_t(x): 
            return grad(t, x, A, b, mu)
        x_min_t = minimize(fun_t, x0, method = 'BFGS', jac = jac_t)['x']
        mins.append(x_min_t)
    mins = np.vstack(mins)
    return mins

def sgd(x_0, eta, A, b, mu, sigma):
    """
    Run stochastic gradient descent with fixed step size
    Inputs:
      - x_0: starting point (d-dimensional vector)
      - eta: step-size (a constant)
      - A: (n,d) design matrix defining LR objectives
      - b: (T+1,n)-array of n-dimensional label vectors
      - mu: regularization parameter
      - sigma^2: L2 variance bound on gradient noise
    Output:
      - x_vals: (T+1,d)-array of estimated x's at each iteration,
                with the most recent values in the last row
    """
    d = x_0.size
    T = b.shape[0] - 1
    t = 0
    x = x_0
    x_vals = [x]
    while t < T:
#         g = grad(t, x, A, b, mu) + noiseball(d, sigma)
        g = samplegrad(t, x, A, b, mu) 
        x = x - eta*g
        x_vals.append(x)
        t += 1
    return np.squeeze(np.array(x_vals))

def avgiter(x, rho): 
    """
    Compute running average of iterates via generic averaging scheme with weight rho
    Inputs:
      - x: (T+1,d)-array of d-dimensional iterates
      - rho: averaging weight
    Output:
      - (T+1,d)-array d-dimentional average iterates
    """
    T = x.shape[0] - 1
    y = [x[0]]
    for i in range(1,T+1):
        z = (1 - rho)*y[i-1] + rho*x[i]
        y.append(z)
    return np.squeeze(np.array(y))

def trckerr1(x, x_min):  
    """
    Compute (T+1)-array of square-L2-distance errors
    Inputs:
      - x: (T+1,d)-array of d-dimensional iterates
      - x_min: (T+1,d)-array of d-dimensional minimizers
    Output:
      - (T+1)-array of square-L2-distance errors
    """
    return norm(x - x_min, axis = 1)**2

def trckerr2(y, A, b, mu, x_min): 
    """
    Compute (T+1)-array of function-value errors
    Inputs:
      - y: (T+1,d)-array of d-dimensional iterates
      - A: (n,d) design matrix defining LR objectives
      - b: (T+1,n)-array of n-dimensional label vectors
      - mu: regularization parameter
      - x_min: (T+1,d)-array of d-dimensional minimizers of LR objectives 
    Output:
      - (T+1)-array of function-value errors  
    """
    T = b.shape[0] - 1
    err = []
    for t in range(0,T+1):
        z = obj(t, y[t], A, b, mu) - obj(t, x_min[t], A, b, mu)
        err.append(z)
    return np.array(err)

def opterr1(mu, eta, x_0, x_init, T):
    """
    Compute (T+1)-array of optimization errors corresponding to trckerr1
    Inputs:
      - mu: regularization parameter
      - eta: SGD step-size (a constant)
      - x_0: starting point of SGD (d-dimensional vector)
      - x_init: initial minimizer
      - T: number of SGD steps
    Output:
      - (T+1)-array of optimization errors corresponding to trckerr1
    """
    r = 1 - mu*eta
    t = np.linspace(0,T,T+1)
    return (r**t)*(norm(x_0 - x_init)**2)

def opterr2(mu, eta, x_0, A, b, x_min):
    """
    Compute (T+1)-array of optimization errors corresponding to trckerr2
    Inputs:
      - mu: regularization parameter
      - eta: SGD step-size (a constant)
      - x_0: starting point of SGD (d-dimensional vector)
      - A: (n,d) matrix determining the LR objectives
      - b: (T+1,n)-array of n-dimensional label vectors defining LR objectives
      - x_min: (T+1,d)-array of d-dimensional minimizers of LR objectives
    Output:
      - (T+1)-array of optimization errors corresponding to trckerr2
    """
    rho = (mu*eta)/(2-(mu*eta))
    r = 1 - rho
    T = x_min.shape[0] - 1
    t = np.linspace(0,T,T+1)
    y = np.tile(x_0, (T+1,1))
    return (r**t)*(trckerr2(y, A, b, mu, x_min) + (mu/4)*(norm(y - x_min, axis = 1)**2))

def empopterr2bd(mu, eta, x_0, A, b_init, x_init, T):
    """
    Compute (T+1)-array of optimization error bounds corresponding to trckerr2
    Inputs:
      - mu: regularization parameter
      - eta: SGD step-size (a constant)
      - x_0: starting point of SGD (d-dimensional vector)
      - A: (n,d) matrix determining the LR objectives
      - b_init: initial label vector in {0,1}^n
      - x_init: minimizer of the initial LR objective
      - T: number of SGD steps
    """
    r = 1 - mu*eta/2
    t = np.linspace(0,T,T+1)
    n = A.shape[0]
    Ax_0 = np.dot(A, x_0)
    Ax_init = np.dot(A, x_init)
    err0 = (1/n)*(np.sum(np.log(1 + np.exp(Ax_0))) - np.dot(b_init, Ax_0)) + (mu/2)*(norm(x_0)**2) - ((1/n)*(np.sum(np.log(1 + np.exp(Ax_init))) - np.dot(b_init, Ax_init)) + (mu/2)*(norm(x_init)**2))
    return (r**t)*err0
    
def trckerr1trials(T, b_init, mu, A, x_0, eta, sigma, N):
    """
    Compute trckerr1 over N trials
    Inputs:
      - T: number of SGD steps
      - b_init: initial label in {0,1}^n, used to generate subsequent labels b via labels(b_init, T)
      - mu: regularization parameter of each LR objective
      - A: (n,d) matrix determining the LR objectives
      - x_0: starting point of each SGD run (d-dimensional vector)
      - eta: SGD step-size (a constant)
      - sigma: gradient L2 variance bound
      - N: number of trials to run
    Output:
      - (T+1,N)-array with t^th row comprised of N trial trackerr1 values
    """  
    err = []
    for i in range(0,N):
        b = labels(b_init, T)
        x_min = mins(A, b, mu)
        x_sgd = sgd(x_0, eta, A, b, mu, sigma)
        err.append(trckerr1(x_sgd, x_min))
    err = np.vstack(err).T
    return err

def empavgtrckerr1(T, b_init, mu, A, x_0, eta, sigma, N): 
    """
    Compute empirical average of trckerr1 over N trials
    Inputs:
      - T: number of SGD steps
      - b_init: initial label in {0,1}^n, used to generate subsequent labels b via labels(b_init, T)
      - mu: regularization parameter of each LR objective
      - A: (n,d) matrix determining the LR objectives
      - x_0: starting point of each SGD run (d-dimensional vector)
      - eta: SGD step-size (a constant)
      - sigma: gradient L2 variance bound
      - N: number of trials to run
    Output:
      - (T+1)-array of average trackerr1 values
    """
    err = trckerr1trials(T, b_init, mu, A, x_0, eta, sigma, N)
    return np.mean(err, axis = 1)
     
def trckerr2trials(T, b_init, mu, A, x_0, eta, sigma, N): 
    """
    Compute trckerr2 over N trials
    Inputs:
      - T: number of SGD steps
      - b_init: initial label in {0,1}^n, used to generate subsequent labels b via labels(b_init, T)
      - mu: regularization parameter of each LR objective
      - A: (n,d) matrix determining the LR objectives
      - x_0: starting point of each SGD run (d-dimensional vector)
      - eta: SGD step-size (a constant)
      - sigma: gradient L2 variance bound
      - N: number of trials to run
    Output:
      - (T+1,N)-array with t^th row comprised of N trial trackerr2 values
    """ 
    rho = (mu*eta)/(2-(mu*eta))
    err = []
    for i in range(0,N):
        b = labels(b_init, T)
        x_min = mins(A, b, mu)
        x_sgd = sgd(x_0, eta, A, b, mu, sigma)
        x_avg = avgiter(x_sgd,rho)
        err.append(trckerr2(x_avg, A, b, mu, x_min))
    err = np.vstack(err).T
    return err
        
def empavgtrckerr2(T, b_init, mu, A, x_0, eta, sigma, N): 
    """
    Compute empirical average of trckerr2 over N trials
    Inputs:
      - T: number of SGD steps
      - b_init: initial label in {0,1}^n, used to generate subsequent labels b via labels(b_init, T)
      - mu: regularization parameter of each LR objective
      - A: (n,d) matrix determining the LR objectives
      - x_0: starting point of each SGD run (d-dimensional vector)
      - eta: SGD step-size (a constant)
      - sigma: gradient L2 variance bound
      - N: number of trials to run
    Output:
      - (T+1)-array of average trackerr2 values
    """
    err = trckerr2trials(T, b_init, mu, A, x_0, eta, sigma, N)
    return np.mean(err, axis = 1)

def opterr2trials(T, b_init, mu, A, x_0, eta, N): 
    """
    Compute opterr2 over N trials
    Inputs:
      - T: number of SGD steps
      - b_init: initial label in {0,1}^n, used to generate subsequent labels b via labels(b_init, T)
      - mu: regularization parameter of each LR objective
      - A: (n,d) matrix determining the LR objectives
      - x_0: starting point of each SGD run (d-dimensional vector)
      - eta: SGD step-size (a constant)
      - N: number of trials to run
    Output:
      - (T+1,N)-array with t^th row comprised of N trial opterr2 values
    """
    err = []
    for i in range(0,N):
        b = labels(b_init, T)
        x_min = mins(A, b, mu)
        err.append(opterr2(mu, eta, x_0, A, b, x_min))
    err = np.vstack(err).T
    return err

def empopterr2(T, b_init, mu, A, x_0, eta, N): 
    """
    Compute empirical average of opterr2 over N trials
    Inputs:
      - T: number of SGD steps
      - b_init: initial label in {0,1}^n, used to generate subsequent labels b via labels(b_init, T)
      - mu: regularization parameter of each LR objective
      - A: (n,d) matrix defining the LR objectives
      - x_0: starting point of each SGD run (d-dimensional vector)
      - eta: SGD step-size (a constant)
      - N: number of trials to run
    Output:
      - (T+1)-array of average opterr2 values
    """
    err = opterr2trials(T, b_init, mu, A, x_0, eta, N)
    return np.mean(err, axis = 1)

def asymptbd1(mu, eta, sigma, Delta):
    """
    Compute asymptotic bound on expected square-L2-distance error
    Inputs:
      - mu: strong convexity parameter of objectives
      - eta: SGD step-size (a constant in (0, 1/(2*L)])
      - sigma: expected gradient L2 variance bound
      - Delta: upper bound on L2 distance between consecutive minimizers
    Output:
      - scalar asymptotic bound on expected square-L2-distance error
    """
    return (1/mu)*(eta*sigma**2 + (Delta**2)/(mu*eta**2))

def asymptbd2(mu, eta, sigma, B):
    """
    Compute asymptotic bound on expected function value error
    Inputs:
      - mu: strong convexity parameter of objectives
      - eta: SGD step-size (a constant in (0, 1/(2*L)])
      - sigma: expected gradient L2 variance bound
      - B: upper bound on gradient drift
    Output:
      - scalar asymptotic bound on expected function value error
    """
    return eta*sigma**2 + (B**2)/((mu**3)*(eta**2))

def eta_opt1(mu, L, sigma, Delta):
    """
    Compute constant SGD step-size optimal for asymptbd1
    Inputs:
      - mu: strong convexity parameter of objectives
      - L: smoothness parameter of objectives
      - sigma: gradient sample L2 variance bound
      - Delta: upper bound on L2 distance between consecutive minimizers (positive scalar)
    """
    return min(1/(2*L), np.power((2*Delta**2)/(mu*sigma**2), 1/3))

def eta_opt2(mu, L, sigma, B):
    """
    Compute constant SGD step-size optimal for asymptbd2
    Inputs:
      - mu: strong convexity parameter of objectives
      - L: smoothness parameter of objectives
      - sigma: gradient sample L2 variance bound
      - B: upper bound on gradient drift (positive scalar)
    """
    return min(1/(2*L), np.power((2*B**2)/((mu**3)*(sigma**2)), 1/3))

