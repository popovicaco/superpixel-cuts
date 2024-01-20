import numpy as np

def pixel_nnls(endmember_spectra, pixel_vec):
    '''
    Description:
        Non-negative Least Squares using Langrangian Method of Multipliers
    ===========================================
    Parameters:
        endmember_spectra  - (nb, ne) NumPy Spectral Endmember Matrix
        pixel_vec          - (1, nb) NumPy Pixel Vector
    ===========================================
    Returns:
        abundance_vec      -  (1, n_end) NumPy Abundance Vector
    ===========================================
    References:
        [1] Lawson, C. L. and R. J. Hanson. Solving Least-Squares Problems.
            Upper Saddle River, NJ: Prentice Hall. 1974. Chapter 23, p. 161.
    '''
    pixel_vec = pixel_vec.reshape((-1, 1))
    # Get the number of end members
    ne = endmember_spectra.shape[1]
    # Initialize set of nonactive columns to null
    P = np.full((ne, 1), False)
    # Initialize set of active columns to all and the initial point x0 to zeros
    Z = np.full((ne, 1), True)
    abundance_vec = np.zeros(shape=(ne, 1))

    # Calculating residuals
    residual = pixel_vec - np.dot(endmember_spectra, abundance_vec)
    w = np.dot(endmember_spectra.T, residual)

    # Tolerance Calculations
    eps = 2.22e-16
    tolerance = 10 * eps * np.linalg.norm(endmember_spectra, 1) * (max(endmember_spectra.shape))  # Why? Nevermind

    # Iteration Criterion
    outer_iteration = 0
    iteration = 0
    max_iteration = 3 * ne
    while np.any(Z) and np.any(w[Z] > tolerance):
        outer_iteration += 1
        # Reset intermediate solution
        z = np.zeros(shape=(ne, 1))
        # Create wz a Lagrange multiplier vector of variables in the zero set
        wz = np.zeros(shape=(ne, 1))
        wz[P] = np.NINF
        wz[Z] = w[Z]
        # Find index of variable with largest Lagrange multiplier, move it from zero set to positive set
        idx = wz.argmax()
        P[idx] = True
        Z[idx] = False
        # Compute the Intermediate Solution using only positive variables in P
        z[P.flatten()] = np.dot(np.linalg.pinv(endmember_spectra[:, P.flatten()]), pixel_vec)
        # Inner loop to remove elements from the positive set that do not belong
        while np.any(z[P] <= 0):
            iteration += 1
            if iteration > max_iteration:
                abundance_vec = z
                residual = pixel_vec - np.dot(endmember_spectra, abundance_vec)
                w = np.dot(endmember_spectra.T, residual)
                return abundance_vec
            # Find indices where approximate solution is negative
            Q = (z <= 0) & P
            # Choose new x subject keeping it non-negative
            alpha = min(abundance_vec[Q] / (abundance_vec[Q] - z[Q]))
            abundance_vec = abundance_vec + alpha * (z - abundance_vec)
            # Reset Z and P given intermediate values of x
            Z = ((abs(abundance_vec) < tolerance) & P) | Z
            P = ~Z
            z = np.zeros(shape=(ne, 1))
            # Resolve for z
            z[P.flatten()] = np.dot(np.linalg.pinv(endmember_spectra[:, P.flatten()]), pixel_vec)
        abundance_vec = np.copy(z)
        residual = pixel_vec - np.dot(endmember_spectra, abundance_vec)
        w = np.dot(endmember_spectra.T, residual)
    return abundance_vec

def asc_preperation(endmember_spectra,data,delta):
    '''
    Description:
        Prepares the ASC Constraint for Matrices
    ===========================================
    Parameters:
        endmember_spectra      - (nb, ne) Known Endmember Spectra Matrix
        data                   - (np, nb) Hyperspectral Image Matrix
    ===========================================
    Returns:
        asc_endmember_spectra  - (nb, ne) ASC Endmember Spectra Matrix
        asc_data               - (np, nb) ASC Hyperspectral Image Matrix
    ===========================================
    References:
        [1] Lawson, C. L. and R. J. Hanson. Solving Least-Squares Problems.
            Upper Saddle River, NJ: Prentice Hall. 1974. Chapter 23, p. 161.
    '''
    asc_endmember_spectra = delta*np.ones(shape=(endmember_spectra.shape[0] + 1, endmember_spectra.shape[1]))
    asc_endmember_spectra[0:endmember_spectra.shape[0],:] = endmember_spectra
    asc_data = delta*np.ones(shape=(data.shape[0] + 1, data.shape[1]))
    asc_data[0:data.shape[0],:] = data
    return asc_endmember_spectra, asc_data

def active_set_fcls(endmember_spectra, data, delta = 100):
    '''
    Description:
        Non-negative Constrained Least Squares using Langrangian Method of Multipliers
    ===========================================
    Parameters:
        endmember_spectra      - (nb, ne) Known Endmember Spectra Matrix
        data                   - (np, nb) Hyperspectral Image Matrix
    ===========================================
    Returns:
        abundance_vec      -  (1, n_end) NumPy Abundance Vector
    ===========================================
    References:
        [1] Lawson, C. L. and R. J. Hanson. Solving Least-Squares Problems.
            Upper Saddle River, NJ: Prentice Hall. 1974. Chapter 23, p. 161.
    '''
    n_bands, n_p = data.shape
    n_end = endmember_spectra.shape[1]
    delta_M, delta_X = asc_preperation(endmember_spectra, data, delta)
    # Allocates space for abundance cube
    abund_mtx = np.zeros(shape=(n_end, n_p))
    for i in range(abund_mtx.shape[1]):
        abund_mtx[:,i] = pixel_nnls(delta_M, delta_X[:,i]).reshape(n_end)
    return abund_mtx

## Fully Constrained Least Squares using ADMM (Method 2)
def convex_projection(X, projection="probability_simplex"):
    '''
        Projects the columns of a matrix X - (m,n) onto a convex set
        projection = 'probability_simplex' will project the columns
        of the matrix onto the set △(N) = { a ∈ R^N_+ | 1^T a = 1 }
        sourced from: https://gist.github.com/mblondel/6f3b7aaad90606b98f71
    '''
    if projection == 'probability_simplex':
        p, n = X.shape
        u = np.sort(X, axis=0)[::-1, ...]
        pi = np.cumsum(u, axis=0) - 1
        ind = (np.arange(p) + 1).reshape(-1, 1)
        mask = (u - pi / ind) > 0
        rho = p - 1 - np.argmax(mask[::-1, ...], axis=0)
        theta = pi[tuple([rho, np.arange(n)])] / (rho + 1)
        return np.maximum(X - theta, 0)    
    
def calculate_norm(data, p = 2, q = 2):
    '''Calculates the L_{p,q} mixed matrix norm'''
    n_rows, n_cols = data.shape
    # sum by row, then column
    return ((data**p).sum(axis = 1)**(q/p)).sum()**(1/q)

def fclsu_admm(M, X, mu = 1, n_iters = 200, eps_tol = 0.001):
    '''
        ADMM Based Impementation of Fully Constrained Linear Unmixing
        Parameters:
            M - Known Endmember Spectra Matrix
            X - Hyperspectral Image Matrix
            mu - ADMM convergence parameter
            eps_tol - convergence criteria
    '''
    history = {
        'loss':[],
        'primal_residual':[],
        'dual_residual':[],
        'n_iters':0
    }
    _, n_endmember = M.shape
    n_p = X.shape[1]

    A = np.ones((n_endmember, n_p))
    B = np.ones((n_endmember, n_p))
    tilde_B = np.ones((n_endmember, n_p))
    tilde_B_prev = np.ones((n_endmember, n_p))

    PI = np.linalg.inv(M.T @ M + mu*np.eye(n_endmember)) #Calculate before hand

    for k in range(n_iters):
        A = PI@(M.T@X + mu*(B-tilde_B)) #A Update
        B = convex_projection(A + tilde_B)      #B Update
        tilde_B_prev = tilde_B.copy()   #Used for tracking
        tilde_B = tilde_B + A - B       #Residual Update

        ## REPORTING ##
        history['loss'].append(calculate_norm(M@A - X)) 
        history['primal_residual'].append(calculate_norm(A-B))
        history['dual_residual'].append(mu*calculate_norm(tilde_B-tilde_B_prev))
        history['n_iters'] += 1

        if calculate_norm(A-B) < eps_tol:
            print(f'Terminated at Iteration {k}')
            break

    abund_mtx = A.copy()
    return abund_mtx, history