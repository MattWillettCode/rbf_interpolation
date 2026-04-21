import numpy as np
import pandas as pd
from helper_functions import (points_in_unit_ball, step3, step3_bounded, step4, step5, step8,
                               mq, gaussian, inv_mq, phi_maker, inv_quadratic, thin_plate_spline,
                               gen_mq, RBF_REGISTRY)
import time
from woodbury_preconditioner import IdentityPreconditioner, WoodburyPreconditioner, MPBulkChebyshevPreconditioner

def FGP(data: list,                       # List of data centers for interpolation
        values: list,                     # List of associated values for each center
        c: float = 1,                   # Shape parameter for your RBF
        q: int = 30,                      # Parameter for the FGP algorithm
        error: float = 1e-5,                     
        seed: int = 42,
        max_iterations: int = 1000,
        rbf_function = mq,                # See below for list of possible inputs.
        verbose = True,
        early_stopping = 10,            # Terminates if error increases for this many iterations in a row.
        first_guess = False,
        proj_CG = False,
        bounded = False,                 # If True, uses step3_bounded so each point appears in at most q lsets.
        hybrid_approach = False,
        p_1 = False,
        p_cheby = False,
        variable = 10                    # variable bounds the size of each lset.
        ):

    """
    Runs the Faul–Goodsell–Powell (FGP) algorithm to build an RBF interpolant.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_dims)
        Interpolation centres.
    values : array-like of shape (n_samples,)
        Target values at the centres.
    c : float, default=0.1
        Shape parameter for the chosen RBF.
    q : int, default=30
        Local neighbourhood size used by FGP (typically 20–50).
    error : float, default=1e-5
        Stopping tolerance on the maximum absolute residual at the centres.
    seed : int, default=42
        Random seed for reproducibility.
    max_iterations : int, default=1000
        Hard cap on iterations.
    rbf_function : callable, default=mq
        rbfs included are: mq, gaussian, inv_mq and inv_quadratic

    Returns
    -------
       helpful_stats : pandas.DataFrame
        Per-iteration diagnostics
          'iteration': num_iterations
          'interp_coeffs': lambdas
          'interp_constant':alpha
          'error': err
          'qvalue': q
          'time_taken': time_taken
    phi : np.array
        The interpolation matrix
    bool: boolean
        True if the algorithm converged within max_iterations iterations
    """

    # -------------------- SETUP -------------------------

    # Reproducibility
    rng = np.random.default_rng(seed)

    # Setup depending on if the interpolation matrix is SPD or not. 
    if rbf_function in [gaussian, inv_mq, inv_quadratic, thin_plate_spline]:
        pos_def_const = -1
    elif rbf_function in [mq, gen_mq]:
        pos_def_const = 1
    else:
        raise ValueError('Make sure your rbf_function is one of [gaussian, mq, inv_mq, inv_quadratic]')

    x_i = np.array(data)
    f_i = pos_def_const * np.array(values)
    n, d = x_i.shape

    # Checking that your data is the right size
    if n != len(f_i):
        raise ValueError('You must have the same number of centers and associated function values')


    # Randomly shuffling the order of the datapoints to avoid combinations that result in slow convergence.
    omeg = rng.permutation(n) + 1
    data = []

    sq_norm = np.sum(x_i**2, axis=1)
    squared_distance_matrix = np.maximum(sq_norm[:, None] + sq_norm[None, :] - 2 * np.dot(x_i, x_i.T), 0.0)
    phi = pos_def_const * rbf_function(squared_distance_matrix, c)

    if np.allclose(phi, np.diag(np.diag(phi))):
        raise ValueError('Phi is diagonal, check your value of c unless this was your intention')

    start_time = time.time()

    # Look up derivative and confirm preconditioner support for this RBF
    _rbf_name = {v[0]: k for k, v in RBF_REGISTRY.items()}.get(rbf_function)
    _rbf_dash  = RBF_REGISTRY[_rbf_name][1] if _rbf_name else None

    woodbury_active = False
    if proj_CG and _rbf_name is not None:
        try:
            if p_1:
                woodbury_precon = WoodburyPreconditioner(
                    x_i, c, rbf_func=rbf_function, rbf_dash=_rbf_dash,
                    pos_def_const=pos_def_const)
            elif p_cheby:
                woodbury_precon = MPBulkChebyshevPreconditioner(
                    x_i, c, rbf_func=rbf_function, rbf_dash=_rbf_dash,
                    pos_def_const=pos_def_const)
            else:
                woodbury_precon = IdentityPreconditioner()
            woodbury_active = True
        except np.linalg.LinAlgError as e:
            print(f'Preconditioner setup failed: {e}')
            return pd.DataFrame(), phi, False, lambdas
    elif proj_CG:
        print(f'Warning: proj_CG is not supported for {rbf_function.__name__}, falling back to standard FGP.')
        woodbury_precon = IdentityPreconditioner()

    # Set up intital values for the interpolation coefficients and interpolation constant, lambdas and alpha
    lambdas = np.zeros(n)
    if first_guess:
        alpha_d   = 2 * d / (d + 2)
        # Use effective (pos_def_const-scaled) RBF values so the approximation
        # matches the SPD matrix phi = pos_def_const * rbf_function(D², c).
        phi_alpha = pos_def_const * rbf_function(alpha_d, c)
        f0        = pos_def_const * rbf_function(0, c)
        b = phi_alpha - f0
        a = phi_alpha / (b * (f0 + (n - 1) * phi_alpha))
        lambdas = a * np.sum(f_i) * np.ones((n)) - 1/b * f_i
        lambdas = lambdas - lambdas.mean() * np.ones(n)

        residual = f_i - phi @ lambdas
        alpha = residual.mean()
        residual = residual - alpha * np.ones(n)
    
    else:
        alpha = 0.5 * (np.max(f_i) + np.min(f_i))
        # Calculating the initial residual.
        residual = f_i - np.ones(n) * alpha

    if hybrid_approach or not woodbury_active:
        distance_matrix = np.sqrt(squared_distance_matrix)
        if bounded:
            membership_count = np.zeros(n + 1, dtype=int)  # 1-indexed; tracks lset appearances per point
        for m in range(1, n):

            if bounded:
                newomeg, lset, lvalue = step3_bounded(omeg, distance_matrix, q, m, n, membership_count, variable)
            else:
                newomeg, lset, lvalue = step3(omeg, distance_matrix, q, m, n)
            omeg = newomeg
            data.append([lvalue, lset, step4(lset,
                                                x_i,
                                                c,
                                                rbf_function,
                                                pos_def_const)])

        data = sorted(data, key=lambda x: x[0])

    # -------------------- SETUP COMPLETE -------------------------
    # -------------------- ITERATION BEGINS -------------------------

    if verbose:
        print('Setup complete, beginning loop')

    # Iteration counter
    num_iterations = 0
    time_taken = 0.0

    # Early stopping if the error starts to diverge
    error_increase_count = 0

    # First error value after setup
    err = np.max(np.abs(residual))

    # Required for iteration
    prev_direction = None

    # Tracking various helpful things across iterations
    helpful_stats = []
    
    setup_time = time.time() - start_time
    time_taken = 0.0
    # Iteration
    while err > error:

        num_iterations += 1

        if hybrid_approach:
            if num_iterations !=2 :
                tau = np.sum(np.array([step5(dat[1], dat[2], residual) for dat in data]), axis = 0)
            if num_iterations == 2:
                tau = woodbury_precon.apply(residual)
                tau = tau - tau.mean() * np.ones(n)

        elif woodbury_active:
            tau = woodbury_precon.apply(residual)
            tau = tau - tau.mean() * np.ones(n)

        else:
            # Generate tau via local cardinal functions first
            tau = np.sum(np.array([step5(dat[1], dat[2], residual) for dat in data]), axis = 0)

        if num_iterations == 1:
            delta = tau.copy()

        else:
            denom = np.dot(delta, prev_direction)
            if denom == 0:
                print('division by 0 error for beta, padding denominator')
                beta = np.dot(tau, prev_direction) / (denom + 1e-10)
            else:
                beta = np.dot(tau, prev_direction) / denom
            delta = tau - beta * delta

        direction = phi @ delta
        prev_direction = direction.copy()

        prev_err = err 
        lambdas, alpha, residual, err = step8(lambdas, alpha, residual, direction, delta)

        if prev_err <= err:
            error_increase_count +=1
        else:
            error_increase_count = 0

        # Early stopping if the error starts to diverge
        if error_increase_count >= early_stopping:
            print(f'{rbf_function}The algorithm did not converge within {num_iterations} iterations - the error kept increasing. Error: {err}.')
            return pd.DataFrame(helpful_stats), phi, False, lambdas

        if verbose:
            print(f'k = {num_iterations} and the error = {err}')
        if np.isnan(err):
            print('Error is NaN, looks like the algorithm has diverged')
            break

        # Tracking error values and alpha
        time_taken = time.time() - start_time
        helpful_stats.append({'iteration': num_iterations, 'interp_coeffs': lambdas, 'interp_constant':alpha, 'error': err, 'qvalue': q, 'time_taken': time_taken, 'setup_time': setup_time})

        if num_iterations > max_iterations:
            print(f'{rbf_function}The algorithm did not converge within {max_iterations} iterations - tune your parameters. Error: {err}.')
            return pd.DataFrame(helpful_stats), phi, False, lambdas
        
    #
    print(f" Time taken: {time_taken} seconds in {num_iterations} iterations, q = {q}, c = {c}, n = {n}")

    helpful_stats = pd.DataFrame(helpful_stats)
    print(lambdas[0])
    return helpful_stats, phi, True, lambdas


#--------------- DEMO -------------------
if __name__ == '__main__':

    from helper_functions import points_in_unit_ball, mq, franke, points_normal, thin_plate_spline, gaussian,inv_mq,inv_quadratic

    n = 2000
    q = 30
    c = 1e-5

    d = 3

    rng = np.random.default_rng(42)
    data = points_in_unit_ball(n, d, 42)
    error = 1e-5
    values = (1+np.exp(-np.abs(data.sum(axis=1))))**-1

    helpful_stats, phi, _, lambdas = FGP(data,
                                         values,
                                         c,
                                         q,
                                         error,
                                         rbf_function = mq,
                                         proj_CG = True,
                                         p_1 = True,
                                         hybrid_approach = True)
    
    helpful_stats, phi, _, lambdas = FGP(data,
                                         values,
                                         c,
                                         q,
                                         error,
                                         rbf_function = mq,
                                         proj_CG = False,
                                         first_guess = False
)
