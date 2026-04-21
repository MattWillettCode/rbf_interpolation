import numpy as np
import pandas as pd
from helper_functions import points_in_unit_ball, step3, step3_bounded, step4, step5, step8, mq, gaussian, inv_mq, inv_quadratic, thin_plate_spline, gen_mq
import time


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
        bounded = False,                 # If True, uses step3_bounded so each point appears in at most q lsets.
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

    start_time = time.time()

    # Set up intital values for the interpolation coefficients and interpolation constant, lambdas and alpha
    lambdas = np.zeros(n)

    alpha = 0.5 * (np.max(f_i) + np.min(f_i))

    # Calculating the initial residual.
    residual = f_i - np.ones(n) * alpha

    # Randomly shuffling the order of the datapoints to avoid combinations that result in slow convergence.
    omeg = rng.permutation(n) + 1
    data = []

    sq_norm = np.sum(x_i**2, axis=1)
    squared_distance_matrix = np.maximum(sq_norm[:, None] + sq_norm[None, :] - 2 * np.dot(x_i, x_i.T), 0.0)
    phi = pos_def_const * rbf_function(squared_distance_matrix, c)
    distance_matrix = np.sqrt(squared_distance_matrix)

    # step3 returns the 'lsets' - approximations for the q nearest neighbours for all but one point.
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

    # Early stopping if the error starts to diverge
    error_increase_count = 0

    # First error value after setup
    err = np.max(np.abs(residual))

    # Required for iteration
    prev_direction = None

    # Tracking various helpful things across iterations
    helpful_stats = []
    # Iteration
    while err > error:

        num_iterations += 1
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
            return pd.DataFrame(helpful_stats), phi, False

        if verbose:
            print(f'k = {num_iterations} and the error = {err}')
        if np.isnan(err):
            print('Error is NaN, looks like the algorithm has diverged')
            break

        # Tracking error values and alpha
        time_taken = time.time() - start_time
        helpful_stats.append({'iteration': num_iterations, 'interp_coeffs': lambdas, 'interp_constant':alpha, 'error': err, 'qvalue': q, 'time_taken': time_taken})

        if num_iterations > max_iterations:
            print(f'{rbf_function}The algorithm did not converge within {max_iterations} iterations - tune your parameters. Error: {err}.')
            return pd.DataFrame(helpful_stats), phi, False
        
    #
    print(f" Time taken: {time_taken} seconds in {num_iterations} iterations, q = {q}, c = {c}, n = {n}")

    helpful_stats = pd.DataFrame(helpful_stats)
    return helpful_stats, phi, True


#--------------- DEMO -------------------
if __name__ == '__main__':

    from helper_functions import points_in_unit_ball, mq, franke, points_normal, thin_plate_spline

    n = 4000
    q = 20
    c = 0
    d = 2

    rng = np.random.default_rng(42)
    data = points_in_unit_ball(n, d, 42)

    values = (1+np.exp(-np.abs(data.sum(axis=1))))**-1
    #mq, gaussian, inv_mq, inv_quadratic, thin_plate_spline
    helpful_stats, phi, _ =  FGP(data,                       # List of data centers for interpolation
                                    values,                     # List of associated values for each center
                                    c,                  # Shape parameter for your RBF
                                    q,                    # Parameter for the FGP algorithm
                                    error = 1e-10,                  
                                    seed = 42,                                 
                                    max_iterations = 500,
                                    rbf_function = mq,                # See below for list of possible inputs.
                                    verbose = True,
                                    )    
