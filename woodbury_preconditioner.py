import numpy as np
from helper_functions import mq, f_dash_mq


class WoodburyPreconditioner:
    """
    Preconditioner for the RBF interpolation matrix Phi based on the
    theoretical limiting approximant M from the high-dimensional theorem.

    Supports mq, inv_mq, gaussian, and inv_quadratic via rbf_func/rbf_dash.

        Phi ≈ beta_2 * I  +  U @ M_core @ U^T

    where:
        - U = [X | 1]  is n x (d+1)
        - M_core is a (d+1) x (d+1) matrix encoding the theoretical limit
        - beta_2 = f(0) - f(alpha_d) + alpha_d * f'(alpha_d)
          (all quantities scaled by pos_def_const so the matrix is SPD)
    """

    def __init__(self, X, c, rbf_func=mq, rbf_dash=f_dash_mq, pos_def_const=1):
        """
        Parameters
        ----------
        X             : np.ndarray, shape (n, d) — data centres
        c             : float — shape parameter
        rbf_func      : callable(r_sq, c) — the RBF
        rbf_dash      : callable(r_sq, c) — derivative of RBF w.r.t. r²
        pos_def_const : +1 (mq/gen_mq) or -1 (inv_mq, gaussian, inv_quadratic)
        """
        n, d = X.shape
        self.n = n
        self.d = d
        self.c = c

        alpha_d = 2 * d / (d + 2)

        # Scale by pos_def_const so all quantities refer to the SPD matrix
        phi_alpha = pos_def_const * rbf_func(alpha_d, c)
        phi_prime = pos_def_const * rbf_dash(alpha_d, c)
        f0        = pos_def_const * rbf_func(0, c)

        beta_2 = f0 - phi_alpha + alpha_d * phi_prime
        self.beta_2 = beta_2

        # Augmented data matrix U = [X | 1], shape (n, d+1)
        self.U = np.hstack([X, np.ones((n, 1))])

        # Small kernel matrix M_core, shape (d+1, d+1)
        M_core = np.zeros((d + 1, d + 1))
        M_core[:d, :d] = -2 * phi_prime * np.eye(d)   # from -2φ' X@X^T
        M_core[d, d]   = phi_alpha                     # from φ_α 1·1^T

        try:
            M_inv = np.linalg.inv(M_core)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(
                f'M_core is singular (phi_prime={phi_prime:.3e}, phi_alpha={phi_alpha:.3e}). '
                f'Try a different value of c.'
            )

        UtU   = self.U.T @ self.U
        inner = M_inv + (1 / beta_2) * UtU
        try:
            self.inner_inv = np.linalg.inv(inner)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(
                f'Woodbury inner matrix (M_inv + U^T U / beta_2) is singular '
                f'(beta_2={beta_2:.3e}). Try a different value of c.'
            )

        self.inv_beta2    = 1.0 / beta_2
        self.inv_beta2_sq = 1.0 / beta_2**2

    def apply(self, v):
        """Apply Phi_approx^{-1} to a vector v."""
        Ut_v       = self.U.T @ v
        correction = self.inner_inv @ Ut_v
        return self.inv_beta2 * v - self.inv_beta2_sq * (self.U @ correction)
    
 

class MPBulkChebyshevPreconditioner:
    """
    Preconditioner for the RBF interpolation matrix Phi using a Chebyshev
    polynomial approximation to the inverse, applied on the bulk spectrum
    predicted by the Marchenko–Pastur law.

    Supports mq, inv_mq, gaussian, and inv_quadratic via rbf_func/rbf_dash.

    On the subspace orthogonal to e, the three-term approximant acts as

        A v = beta_2 * v + beta_3 * X_c @ (X_c^T @ v) / d

    where X_c is the column-centred data matrix and beta_3 = -2 * f'(alpha_d).
    Cost:
        Setup:  O(n d)   — column-centring X
        Apply:  O(k n d) — k matrix-vector products with X_c
    """

    def __init__(self, X, c, rbf_func=mq, rbf_dash=f_dash_mq, pos_def_const=1, degree=4):
        """
        Parameters
        ----------
        X             : np.ndarray, shape (n, d) — data centres
        c             : float — shape parameter
        rbf_func      : callable(r_sq, c) — the RBF
        rbf_dash      : callable(r_sq, c) — derivative of RBF w.r.t. r²
        pos_def_const : +1 (mq/gen_mq) or -1 (inv_mq, gaussian, inv_quadratic)
        degree        : int — degree of Chebyshev polynomial approximation
        """
        n, d = X.shape
        self.n = n
        self.d = d
        self.c = c
        self.degree = degree

        alpha_d   = 2 * d / (d + 2)
        phi_alpha = pos_def_const * rbf_func(alpha_d, c)
        phi_prime = pos_def_const * rbf_dash(alpha_d, c)
        f0        = pos_def_const * rbf_func(0, c)

        beta_2 = f0 - phi_alpha + alpha_d * phi_prime
        beta_3 = -2 * phi_prime

        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.mean_scaling = 1.0 / (beta_2 + n * phi_alpha)

        # Column-centred data matrix
        self.Xc = X - X.mean(axis=0, keepdims=True)

        # Marchenko–Pastur bulk edges for (1/d) X_c^T X_c
        gamma = n / d
        sigma2 = np.sum(self.Xc ** 2) / (n * d)  # empirical variance per entry
        mu_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
        mu_plus  = sigma2 * (1 + np.sqrt(gamma)) ** 2

        # Map to eigenvalues of A = beta_2 I + beta_3 X_c X_c^T / d.
        # beta_3 < 0, so larger mu gives more negative lam: mu_plus -> lam_minus.
        lam_minus = beta_2 + beta_3 * mu_plus
        lam_plus  = beta_2 + beta_3 * mu_minus

        self.lam_minus = lam_minus
        self.lam_plus  = lam_plus

        if lam_plus >= 0:
            raise ValueError(
                "Bulk interval crosses 0; polynomial inverse is unsafe here."
            )

        # Chebyshev interpolation nodes and coefficients for 1/x on [lam_-, lam_+]
        j = np.arange(degree + 1)
        theta = (j + 0.5) * np.pi / (degree + 1)
        t_nodes = np.cos(theta)                     # in [-1, 1]
        x_nodes = (0.5 * (lam_plus - lam_minus) * t_nodes
                    + 0.5 * (lam_plus + lam_minus))
        f_nodes = 1.0 / x_nodes

        coeffs = np.zeros(degree + 1)
        for k in range(degree + 1):
            coeffs[k] = (2.0 / (degree + 1)) * np.sum(
                f_nodes * np.cos(k * theta)
            )
        coeffs[0] *= 0.5
        self.coeffs = coeffs

    def _A_apply(self, v):
        """Apply the bulk operator A = beta_2 I + (beta_3/d) X_c X_c^T."""
        return self.beta_2 * v + self.beta_3 * (self.Xc @ (self.Xc.T @ v)) / self.d

    def _A_tilde_apply(self, v):
        """Apply the affinely shifted operator that maps [lam_-, lam_+] to [-1, 1]."""
        Av = self._A_apply(v)
        c1 = 2.0 / (self.lam_plus - self.lam_minus)
        c2 = (self.lam_plus + self.lam_minus) / (self.lam_plus - self.lam_minus)
        return c1 * Av - c2 * v
    def apply(self, v):
            mean_v = v.mean()
            r = v - mean_v  # project onto e^perp

            v0 = r.copy()
            y = self.coeffs[0] * v0

            if self.degree == 0:
                # --- UPDATED RETURN ---
                return y + (mean_v * self.mean_scaling)

            v1 = self._A_tilde_apply(r)
            y = y + self.coeffs[1] * v1

            for k in range(1, self.degree):
                v2 = 2.0 * self._A_tilde_apply(v1) - v0
                y = y + self.coeffs[k + 1] * v2
                v0, v1 = v1, v2

            # --- UPDATED RETURN ---
            return y + (mean_v * self.mean_scaling)

class IdentityPreconditioner:
    """No-op preconditioner — apply() returns v unchanged."""

    def __init__(self):
        pass

    def apply(self, v):
        return v
