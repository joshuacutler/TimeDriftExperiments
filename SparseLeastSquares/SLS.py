import numpy as np
import math
from numpy.linalg import norm
from scipy.stats import ortho_group
from scipy.optimize import bisect

def matrix(n, d, mu, L):
    """
    Generate a matrix A of size (n,d) with min. sing. val. sqrt(mu) and max sing. val. sqrt(L)
    Inputs:
      - n: number of rows of the matrix A
      - d: number of columns of the matrix A
      - mu: minimal eigenvalue of np.dot(A.T,A)
      - L: maximal eigenvalue of np.dot(A.T,A)
    """
    U = ortho_group.rvs(n)
    diag = np.linspace(np.sqrt(mu),np.sqrt(L),min(n,d))
    S = np.zeros((n,d))
    np.fill_diagonal(S, diag)
    V = ortho_group.rvs(d)
    A = np.dot(U, np.dot(S,V.T))
    return A

def noiseL1(d, r):
    """
    Sample a vector uniformly from the centered d-dimensional L1 ball of radius r
    Inputs:
      - d: dimension of output vector
      - r: radius of sampled L1 ball  
    """
    laplace = np.random.laplace(size=d)
    length = norm(laplace, ord = 1)
    if length == 0.0:
        z = laplace
    else:
        s = np.random.uniform()**(1.0/d)
        z = np.multiply(laplace, r*s/length)
    return z

def noisesphere(d, r):
    """
    Sample a vector uniformly from the centered (d-1)-dimensional L2 sphere of radius r
    Inputs:
      - d: dimension of output vector
      - r: radius of sampled L2 sphere  
    """
    gauss = np.random.normal(size=d)
    length = norm(gauss)
    if length == 0.0:
        z = r*np.ones(d)/norm(np.ones(d))
    else:
        z = np.multiply(gauss, r/length)
    return z

def sparserdwlk(x_init, Delta, T):
    """
    Generate a sparse random walk in unit L1 ball starting at x_init
    Inputs:
      - x_init: starting point of random walk (d-dimensional vector in unit L1 ball) with nonzero entries in at most s coordinate
      - Delta: upper bound on expected L2 distance between consecutive steps, in (0,sqrt(2)]
      - T: number of steps taken from x_init
    Output:
      - Random walk path stored as (T+1,d)-array, with the t^th step comprising the t^th row.  
    """
    d = x_init.size
    s = np.count_nonzero(x_init)
    x = x_init
    path = [x]
    for i in range(0,T):
        if np.random.random() <= (4 - 2*Delta**2)/(4 - Delta**2):
            ind = np.ravel(np.nonzero(x))
            v = np.zeros(d)
            y = 2*np.ones(d)
            while norm(y, ord = 1) > 1:
                u = noisesphere(s, Delta/np.sqrt(2))
                v[ind] = u
                y = x + v
            x = y
        else: 
            ind_nz = np.ravel(np.nonzero(x))
            ind_z = np.ravel(np.nonzero(x == 0))
            j = np.random.choice(ind_nz)
            k = np.random.choice(ind_z)
            x[j], x[k] = x[k], x[j]
        path.append(x.copy())
    path = np.stack(path, axis = 0)
    return path

def sthresh(x, u):
    """
    Apply the u-soft-thresholding operator to x
    Inputs:
      - x: input vector
      - u: soft-thresholding parameter  
    """
    y = np.absolute(x)-u
    y[y<0] = 0
    return np.sign(x)*y

def projL1(x):  
    """
    Compute the orthogonal projection of x onto the the L1 ball
    Inputs:
      - x: input vector 
    """
    if norm(x, ord=1) <= 1:
        z = x
    else:
        def g(u):
            y = np.absolute(x)-u
            y[y<0] = 0
            return np.sum(y) - 1
        u_opt = bisect(g, 0, norm(x, np.inf), xtol = 1e-16)
        z = sthresh(x, u_opt)
    return z

def obj(t, x, A, x_min, C):
    """
    Evaluate the t^th LS objective at x, given design matrix A, minimizer sequence x_min, and observation covariance matrix C
    Inputs:
      - t: time index, range 0 to T
      - x: d-dimensional input vector
      - A: (n,d) design matrix defining LS objectives
      - x_min: (T+1,d)-array of minimizers
      - C: (n,n) observation covariance matrix C defining LS objectives
    """
    return 0.5*(norm(np.dot(A,x) - np.dot(A, x_min[t]))**2) + 0.5*np.trace(C)
    
def grad(t, x, A, x_min): 
    """
    Evaluate the gradient of the t^th LS objective at x, given design matrix A and minimizer sequence x_min
    Inputs:
      - t: time index, range 0 to T
      - x: d-dimensional imput vector
      - A: (n,d) design matrix defining LS objectives
      - x_min: (T+1,d)-array of minimizers
    """
    return np.dot(np.dot(A.T, A), x - x_min[t]) 
    
def sgrad(t, x, A, x_min, C): 
    """
    Sample gradient proxy of the t^th LS objective at x, given design matrix A, minimizer sequence x_min, and observation covariance matrix C
    Inputs:
      - t: time index, range 0 to T
      - x: d-dimensional imput vector
      - A: (n,d) design matrix defining LS objectives
      - x_min: (T+1,d)-array of minimizers
      - C: (n,n) observation covariance matrix C defining LS objectives
    """
    b_t = np.random.multivariate_normal(np.dot(A, x_min[t]), C)
    return np.dot(A.T, np.dot(A,x) - b_t)

def sgd(x_0, eta, A, x_min, C):
    """
    Run proximal stochastic gradient descent (projecting onto the L1 ball) with fixed step size
    Inputs:
      - x_0: starting point (d-dimensional vector)
      - eta: step-size (a constant)
      - A: (n,d) design matrix defining LS objectives
      - x_min: (T+1,d)-array of minimizers defining LS objectives
      - C: (n,n) observation covariance matrix C defining LS objectives
    Output:
      - x_vals: (T+1,d)-array of estimated x's at each iteration,
                with the most recent values in the last row
    """
    
    T = x_min.shape[0] - 1
    t = 0
    x = x_0
    x_vals = [x]
    while t < T:
        g = sgrad(t, x, A, x_min, C)
        x = projL1(x - eta*g)
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

def trckerr2(y, A, x_min): 
    """
    Compute (T+1)-array of function-value errors
    Inputs:
      - y: (T+1,d)-array of d-dimensional iterates
      - A: (n,d) design matrix defining LS objectives
      - x_min: (T+1,d)-array of d-dimensional minimizers defining LS objectives
    Output:
      - (T+1)-array of function-value errors  
    """
    u = np.dot(y, A.T)
    v = np.dot(x_min, A.T)
    return 0.5*(norm(u - v, axis = 1)**2)

def opterr1(mu, eta, x_0, x_init, T):
    """
    Compute (T+1)-array of optimization errors corresponding to trckerr1
    Inputs:
      - mu: strong convexity parameter of objectives
      - eta: SGD step-size (a constant)
      - x_0: starting point of SGD (d-dimensional vector)
      - x_init: starting point of minimizer random walk
      - T: number of SGD steps
    Output:
      - (T+1)-array of optimization errors corresponding to trckerr1
    """
    t = np.linspace(0,T,T+1)
    r = 1 - mu*eta
    return (r**t)*(norm(x_0 - x_init)**2)

def opterr2(mu, eta, x_0, A, x_min):
    """
    Compute (T+1)-array of optimization errors corresponding to trckerr2
    Inputs:
      - mu: strong convexity parameter of objectives
      - eta: SGD step-size (a constant)
      - x_0: starting point of SGD (d-dimensional vector)
      - A: (n,d) design matrix defining LS objectives
      - x_min: (T+1,d)-array of d-dimensional minimizers defining LS objectives
    Output:
      - (T+1)-array of optimization errors corresponding to trckerr2
    """
    rho = (mu*eta)/(2-(mu*eta))
    r = 1 - rho
    T = x_min.shape[0] - 1
    d = x_min.shape[1]
    n = A.shape[0] 
    t = np.linspace(0,T,T+1)
    u = np.reshape(np.dot(A, x_0), (1, n))
    v = np.dot(x_min, A.T)
    w = np.reshape(x_0, (1, d))
    return (r**t)*(0.5*(norm(u - v, axis = 1)**2) + (mu/4)*(norm(w - x_min, axis = 1)**2))

def empopterr2bd(mu, eta, x_0, A, x_init, T):
    """
    Compute (T+1)-array of optimization error bounds corresponding to trckerr2
    Inputs:
      - mu: strong convexity parameter of objectives
      - eta: SGD step-size (a constant)
      - x_0: starting point of SGD (d-dimensional vector)
      - A: (n,d) design matrix defining LS objectives
      - x_init: minimizer of the initial LS objective (d-dimensional vector)
      - T: number of steps
    """
    r = 1 - mu*eta/2
    t = np.linspace(0,T,T+1)
    u = np.dot(A, x_0)
    v = np.dot(A, x_init)
    return (r**t)*0.5*(norm(u - v)**2)

def trckerr1trials(x_init, Delta, T, A, sigma, x_0, eta, N): 
    """
    Compute trckerr1 over N trials using sparserdwlk to generate x_min for each trial
    Inputs:
      - x_init: starting point of each minimizer random walk (d-dimensional vector)
      - Delta: upper bound on L2 distance between consecutive minimizer steps (nonnegative scalar)
      - T: number of minimizer steps taken from x_init (also the number of SGD steps)
      - A: (n,d) design matrix defining LS objectives
      - sigma: expected gradient L2 variance bound, determines with A the trace of the (n,n) observation covariance matrix C defining LS objectives
      - x_0: starting point of each SGD run (d-dimensional vector)
      - eta: SGD step-size (a constant)
      - N: number of trials to run
    Output:
      - (T+1,N)-array with t^th row comprised of N trial trackerr1 values
    """
    n = A.shape[0]
    C = (sigma**2/(n*norm(A,ord=2)**2))*np.identity(n)     # spherical (n,n) covariance matrix with trace leq sigma**2/norm(A,ord=2)**2
    # PD(n, (sigma/norm(A,ord=2))**2)                      # non-spherical (n,n) covariance matrix with trace leq sigma**2/norm(A,ord=2)**2   
    err = []
    for i in range(0,N):
        x_min = sparserdwlk(x_init, Delta, T)
        x_sgd = sgd(x_0, eta, A, x_min, C)
        err.append(trckerr1(x_sgd, x_min))
    err = np.vstack(err).T
    return err

def empavgtrckerr1(x_init, Delta, T, A, sigma, x_0, eta, N): 
    """
    Compute empirical average of trckerr1 over N trials using sparserdwlk to generate x_min for each trial
    Inputs:
      - x_init: starting point of each minimizer random walk (d-dimensional vector)
      - Delta: upper bound on L2 distance between consecutive minimizer steps (nonnegative scalar)
      - T: number of minimizer steps taken from x_init (also the number of SGD steps)
      - A: (n,d) design matrix defining LS objectives
      - sigma: expected gradient L2 variance bound, determines with A the trace of the (n,n) observation covariance matrix C defining LS objectives
      - x_0: starting point of each SGD run (d-dimensional vector)
      - eta: SGD step-size (a constant)
      - N: number of trials to run
    Output:
      - (T+1)-array of average trackerr1 values
    """
    err = trckerr1trials(x_init, Delta, T, A, sigma, x_0, eta, N)
    return np.mean(err, axis = 1)
     
def trckerr2trials(x_init, Delta, T, A, sigma, x_0, eta, mu, N): 
    """
    Compute trckerr2 over N trials using sparserdwlk to generate x_min for each trial
    Inputs:
      - x_init: starting point of each minimizer random walk (d-dimensional vector)
      - Delta: upper bound on L2 distance between consecutive minimizer steps (nonnegative scalar)
      - T: number of minimizer steps taken from x_init (also the number of SGD steps)
      - A: (n,d) design matrix defining LS objectives
      - sigma: expected gradient L2 variance bound, determines with A the trace of the (n,n) observation covariance matrix C defining LS objectives
      - x_0: starting point of each SGD run (d-dimensional vector)
      - eta: SGD step-size (a constant)
      - mu: strong convexity parameter of objectives
      - N: number of trials to run
    Output:
      - (T+1,N)-array with t^th row comprised of N trial trackerr2 values
    """
    n = A.shape[0]
    C = (sigma**2/(n*norm(A,ord=2)**2))*np.identity(n)     # spherical (n,n) covariance matrix with trace leq sigma**2/norm(A,ord=2)**2
    # PD(n, (sigma/norm(A,ord=2))**2)                      # non-spherical (n,n) covariance matrix with trace leq sigma**2/norm(A,ord=2)**2   
    err = []
    for i in range(0,N):
        x_min = sparserdwlk(x_init, Delta, T)
        x_sgd = sgd(x_0, eta, A, x_min, C)
        rho = (mu*eta)/(2-(mu*eta))
        x_avg = avgiter(x_sgd,rho)
        err.append(trckerr2(x_avg, A, x_min))
    err = np.vstack(err).T
    return err  
    
def empavgtrckerr2(x_init, Delta, T, A, sigma, x_0, eta, mu, N): 
    """
    Compute empirical average of trckerr2 over N trials using sparserdwlk to generate x_min for each trial
    Inputs:
      - x_init: starting point of each minimizer random walk (d-dimensional vector)
      - Delta: upper bound on L2 distance between consecutive minimizer steps (nonnegative scalar)
      - T: number of minimizer steps taken from x_init (also the number of SGD steps)
      - A: (n,d) design matrix defining LS objectives
      - sigma: expected gradient L2 variance bound, determines with A the trace of the (n,n) observation covariance matrix C defining LS objectives
      - x_0: starting point of each SGD run (d-dimensional vector)
      - eta: SGD step-size (a constant)
      - mu: strong convexity parameter of objectives
      - N: number of trials to run 
    Output:
      - (T+1)-array of average trackerr2 values
    """
    err = trckerr2trials(x_init, Delta, T, A, sigma, x_0, eta, mu, N)
    return np.mean(err, axis = 1)    

def opterr2trials(x_init, Delta, T, A, x_0, eta, mu, N): 
    """
    Compute opterr2 over N trials using sparserdwlk to generate x_min for each trial
    Inputs:
      - x_init: starting point of each minimizer random walk (d-dimensional vector)
      - Delta: upper bound on L2 distance between consecutive minimizer steps (nonnegative scalar)
      - T: number of minimizer steps taken from x_init (also the number of SGD steps)
      - A: (n,d) design matrix defining LS objectives
      - x_0: starting point of each SGD run (d-dimensional vector)
      - eta: SGD step-size (a constant)
      - mu: strong convexity parameter of objectives
      - N: number of trials to run
    Output:
      - (T+1,N)-array with t^th row comprised of N trial opterr2 values
    """
    err = []
    for i in range(0,N):
        x_min = sparserdwlk(x_init, Delta, T)
        err.append(opterr2(mu, eta, x_0, A, x_min))
    err = np.vstack(err).T
    return err

def empopterr2(x_init, Delta, T, A, x_0, eta, mu, N): 
    """
    Compute empirical average of opterr2 over N trials using sparserdwlk to generate x_min for each trial
    Inputs:
      - x_init: starting point of each minimizer random walk (d-dimensional vector)
      - Delta: upper bound on L2 distance between consecutive minimizer steps (nonnegative scalar)
      - T: number of minimizer steps taken from x_init (also the number of SGD steps)
      - A: (n,d) design matrix defining LS objectives
      - x_0: starting point of each SGD run (d-dimensional vector)
      - eta: SGD step-size (a constant)
      - mu: strong convexity parameter of objectives
      - N: number of trials to run
    Output:
      - (T+1)-array of average opterr2 values
    """
    err = opterr2trials(x_init, Delta, T, A, x_0, eta, mu, N)
    return np.mean(err, axis = 1)

def asymptbd1(mu, eta, sigma, Delta):
    """
    Compute asymptotic bound on expected square-L2-distance error
    Inputs:
      - mu: strong convexity parameter of objectives
      - eta: SGD step-size (a constant in (0, 1/(2*L)])
      - sigma: expected gradient L2 variance bound
      - Delta: upper bound on L2 distance between consecutive minimizer steps
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
      - B: upper bound on consecutive function gaps
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
      - Delta: upper bound on minimizer drift
    """
    return min(1/(2*L), np.power((2*Delta**2)/(mu*sigma**2), 1/3))

def eta_opt2(mu, L, sigma, B):
    """
    Compute constant SGD step-size optimal for asymptbd2
    Inputs:
      - mu: strong convexity parameter of objectives
      - L: smoothness parameter of objectives
      - sigma: gradient sample L2 variance bound
      - B: upper bound on gradient drift
    """
    return min(1/(2*L), np.power((2*B**2)/((mu**3)*(sigma**2)), 1/3))

