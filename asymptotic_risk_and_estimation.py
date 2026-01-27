# PLOT SNR OVER LAMBDA - DETERMINISTIC SIGNAL

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# USER-DEFINED VALUES
n = 400
p = 200
K = 1
num_seeds = 30    # at least 2, number of random seeds to take avg of

# Covariance type selections
sigma_type = "spiked"        # Options: "ar1_rho0.5", "spiked", "identity", "ar1_rho0.25"
sigma_beta_type = "top_aligned"  # Options: "top50", "bottom50", "mixed50", "identity", "top_aligned", "bottom_aligned"

spike_frac = 0.1              # Sparse level
align_frac = 0.9              # Alignment factor

# lambda range
lambda_regs = np.logspace(np.log10(0.01), np.log10(50), 30)
num_lams = len(lambda_regs)

# Seeds
seeds = list(range(2025 - num_seeds // 2, 2025 + num_seeds // 2))  # 10 seeds around 2025, reduce for a faster run

# Pairs of (alpha, sigma)
pairs = [(np.sqrt(0.2), 1.0), (np.sqrt(0.5), 1.0), (1.0, 1.0), (np.sqrt(3), 1.0), (np.sqrt(5), 1.0)]
num_pairs = len(pairs)

def ar1_covariance(p, rho):
    """Generate AR(1) covariance matrix with correlation rho (vectorized)."""
    i = np.arange(p)
    j = np.arange(p)
    return rho ** np.abs(i[:, None] - j[None, :])

def spiked_covariance(p, num_spikes=1, spike_strength=5.0):
    """Generate spiked covariance: identity + low-rank spikes."""
    Sigma = np.eye(p)
    for _ in range(num_spikes):
        v = np.random.randn(p)
        v /= np.linalg.norm(v)
        Sigma += spike_strength * np.outer(v, v)
    return Sigma

def generate_covariances(p, sigma_type, sigma_beta_type, spike_frac=0.1, align_frac=0.9):
    """Generate feature covariance Sigma and signal covariance Sigma_beta.

    For sigma_beta_type='top_aligned' or 'bottom_aligned', uses spike_frac (sparse level)
    and align_frac (alignment factor) to control structure.
    """
    # Feature covariance Sigma
    if sigma_type == "ar1_rho0.5":
        Sigma = ar1_covariance(p, 0.5)
    elif sigma_type == "ar1_rho0.25":
        Sigma = ar1_covariance(p, 0.25)
    elif sigma_type == "spiked":
        Sigma = spiked_covariance(p)
    elif sigma_type == "identity":
        Sigma = np.eye(p)
    else:
        raise ValueError(f"Invalid sigma_type: {sigma_type}")

    # Eigendecomposition for alignment
    eigvals, V = np.linalg.eigh(Sigma)

    # Signal covariance Sigma_beta
    k_top = 50
    k_bottom = 50
    m_amp = 50
    amp_factor = 10.0

    if sigma_beta_type == "top50":
        # Energy on top k eigen-directions
        D = np.ones(p) * 0.05
        D[-k_top:] = 1.0
        Sigma_beta = V @ np.diag(D) @ V.T
        trace = np.trace(Sigma_beta)
        Sigma_beta = (p / trace) * Sigma_beta
    elif sigma_beta_type == "bottom50":
        # Energy on bottom k eigen-directions
        D = np.ones(p) * 0.05
        D[:k_bottom] = 1.0
        Sigma_beta = V @ np.diag(D) @ V.T
        trace = np.trace(Sigma_beta)
        Sigma_beta = (p / trace) * Sigma_beta
    elif sigma_beta_type == "mixed50":
        # Amplified first m coordinates (diagonal)
        D = np.ones(p)
        D[:m_amp] *= amp_factor
        Sigma_beta = np.diag(D)
        trace = np.trace(Sigma_beta)
        Sigma_beta = (p / trace) * Sigma_beta
    elif sigma_beta_type == "identity":
        Sigma_beta = np.eye(p)
    elif sigma_beta_type == "top_aligned":
        k = int(spike_frac * p)
        D = np.zeros(p)
        D[-k:] = align_frac * p / k      # Top k: equal share of aligned energy
        D[:-k] = (1 - align_frac) * p / (p - k)  # Bottom: equal share of unaligned
        Sigma_beta = V @ np.diag(D) @ V.T
    elif sigma_beta_type == "bottom_aligned":
        k = int(spike_frac * p)
        D = np.zeros(p)
        D[:k] = align_frac * p / k        # Bottom k: equal share of aligned energy
        D[k:] = (1 - align_frac) * p / (p - k)  # Top: equal share of unaligned
        Sigma_beta = V @ np.diag(D) @ V.T
    else:
        raise ValueError(f"Invalid sigma_beta_type: {sigma_beta_type}")

    return Sigma, Sigma_beta

def excess_risk(beta_hat, beta_star, Sigma, sigma):
    """Compute out-of-sample excess risk."""
    diff = beta_hat - beta_star
    return diff.T @ Sigma @ diff + sigma**2

def solve_kappa_bisect(lam: float, gamma: float, s: np.ndarray,
                       tol: float = 1e-12, maxit: int = 200) -> float:

    s = np.asarray(s)

    def g(k: float) -> float:
        m1 = np.mean(s/(s+k))
        return k*(1 - gamma*m1) - lam

    lo = 0.0
    hi = max(lam, 1.0)
    gh = g(hi)
    # expand bracket if needed
    for _ in range(100):
        if gh > 0:
            break
        hi *= 2.0
        gh = g(hi)
    if gh <= 0:
        raise RuntimeError("Failed to bracket root for kappa")

    for _ in range(maxit):
        mid = 0.5*(lo+hi)
        gm = g(mid)
        if gm > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol*max(1.0, hi):
            break
    return 0.5*(lo+hi)

def compute_asym_risks(gamma: float, Sigma: np.ndarray, beta_star: np.ndarray, lambdas: np.ndarray, sigma: float) -> dict[str, np.ndarray]:

    evals, V = np.linalg.eigh(Sigma)
    p = len(evals)
    L = len(lambdas)
    lambdas = np.asarray(lambdas)
    I = np.eye(p)

    sigma2 = sigma**2

    # Precompute proj = V.T @ beta_star for efficient computations
    proj = V.T @ beta_star

    # Precompute coefficients
    kappa = np.zeros(L)
    t2 = np.zeros(L); t3 = np.zeros(L); t4 = np.zeros(L)
    b = np.zeros(L); E = np.zeros(L)
    a2 = np.zeros(L); a3 = np.zeros(L); a4 = np.zeros(L)
    U2 = np.zeros(L); U3 = np.zeros(L); U4 = np.zeros(L)

    for i, lam in enumerate(lambdas):
        k = solve_kappa_bisect(float(lam), float(gamma), evals)
        kappa[i] = k
        denom = evals + k
        G = 1.0 / denom
        t2[i] = gamma * np.mean((evals**2) * (G**2))
        t3[i] = gamma * np.mean((evals**2) * (G**3))
        t4[i] = gamma * np.mean((evals**2) * (G**4))
        b[i] = 1.0 / (1.0 - t2[i])
        E[i] = k - b[i] * lam + (b[i]**2) * k * lam * t3[i]
        a4[i] = (b[i]**3) * (k**2) * (lam**2)
        a3[i] = 2 * (b[i]**2) * k * lam * E[i]
        a2[i] = b[i] * (E[i]**2) + (b[i]**4) * (k**2) * (lam**2) * t4[i] + (b[i]**5) * (k**2) * (lam**2) * (t3[i]**2)
        U2[i] = t2[i] / (1 - t2[i])
        U3[i] = t3[i] * (b[i]**3)
        U4[i] = t4[i] * (b[i]**4) + 2 * (t3[i]**2) * (b[i]**5)

    coeffs = dict(kappa=kappa, t2=t2, t3=t3, t4=t4, b=b, E=E, a2=a2, a3=a3, a4=a4, U2=U2, U3=U3, U4=U4)

    # Compute q2, q3, q4 using beta_star
    q2 = np.zeros(L)
    q3 = np.zeros(L)
    q4 = np.zeros(L)
    for i, k in enumerate(kappa):
        denom = evals + k
        g2 = evals / denom**2
        g3 = evals / denom**3
        g4 = evals / denom**4
        q2[i] = np.dot(proj**2, g2)
        q3[i] = np.dot(proj**2, g3)
        q4[i] = np.dot(proj**2, g4)

    # Compute Q2, Q3, Q4
    kappa = coeffs['kappa']; b = coeffs['b']; E = coeffs['E']
    a2 = coeffs['a2']; a3 = coeffs['a3']; a4 = coeffs['a4']
    Q2 = (kappa**2) * b * q2
    Q3 = 2 * kappa * b * E * q2 + 2 * (kappa**2) * (b**2) * lambdas * q3
    Q4 = a2 * q2 + a3 * q3 + a4 * q4

    # Compute components A, B, C, D, xi, R1
    B0 = Q2
    C_bias = 2 * Q2 - 0.5 * Q3
    tildeB0 = 4 * Q2 - 2 * Q3 + Q4

    V0 = sigma2 * U2
    C_var = sigma2 * (U2 - lambdas * U3)
    tildeV0 = sigma2 * (U2 - 2 * lambdas * U3 + (lambdas**2) * U4)

    A = B0 + V0
    B = tildeB0 + tildeV0
    C = C_bias + C_var
    D = A + B - 2 * C

    xi = np.full_like(lambdas, np.nan, dtype=float)
    R1 = np.full_like(lambdas, np.nan, dtype=float)
    mask = D > 1e-14
    xi[mask] = (A[mask] - C[mask]) / D[mask]
    R1[mask] = sigma2 + A[mask] - (A[mask] - C[mask])**2 / D[mask]

    # Compute asymptotics
    asym_R0 = A + sigma2
    asym_Rt0 = B + sigma2
    asym_R1 = R1
    asym_A = A
    asym_B = B
    asym_C = C

    return dict(
        asym_R0=asym_R0, asym_Rt0=asym_Rt0, asym_R1=asym_R1,
        asym_A=asym_A, asym_B=asym_B, asym_C=asym_C
    )


# Precompute Covariances
Sigma, Sigma_beta = generate_covariances(p, sigma_type, sigma_beta_type)
gamma = p / n
I_p = np.eye(p)


#### Compute Results for Each (alpha, sigma) Pair

full_results = {}
for pair in pairs:
    alpha, sigma = pair
    key = pair
    full_results[key] = {}

    # Sample fixed beta_star for this pair
    cov_beta = (alpha**2 / p) * Sigma_beta
    np.random.seed(2025)  # Fixed seed for reproducible beta_star sampling
    beta_star = np.random.multivariate_normal(np.zeros(p), cov_beta)

    # Asymptotic risks over all lambdas
    results = compute_asym_risks(gamma, Sigma, beta_star, lambda_regs, sigma)
    asym_R0 = results['asym_R0']
    asym_Rt0 = results['asym_Rt0']
    asym_R1 = results['asym_R1']
    asym_A = results['asym_A']
    asym_B = results['asym_B']
    asym_C = results['asym_C']

    denom_asym = asym_A + asym_B - 2 * asym_C
    asym_xi = np.zeros(num_lams)
    mask = np.abs(denom_asym) > 1e-12
    asym_xi[mask] = (asym_A[mask] - asym_C[mask]) / denom_asym[mask]

    full_results[key]['asym_R0'] = asym_R0
    full_results[key]['asym_Rt0'] = asym_Rt0
    full_results[key]['asym_R1'] = asym_R1
    full_results[key]['asym_A'] = asym_A
    full_results[key]['asym_B'] = asym_B
    full_results[key]['asym_C'] = asym_C
    full_results[key]['asym_xi'] = asym_xi

    # Optimal ridge baselines
    lambda_opt_ridge = gamma * (sigma**2 / alpha**2)
    all_risk_opt_ridges = []
    all_risk_opt_gen_ridges = []
    all_risk_nulls = []

    for seed in seeds:
        np.random.seed(seed)

        # Generate data (fixed beta_star)
        X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
        noise = np.random.randn(n) * sigma
        y = X @ beta_star + noise

        XtX = X.T @ X

        # Optimal ridge
        beta_opt_ridge = np.linalg.solve(XtX / n + lambda_opt_ridge * I_p, X.T @ y / n)
        risk_opt_ridge = excess_risk(beta_opt_ridge, beta_star, Sigma, sigma)

        # Optimal generalized ridge
        beta_opt_gen_ridge = np.linalg.solve(XtX / n + lambda_opt_ridge * np.linalg.inv(Sigma_beta), X.T @ y / n)
        risk_opt_gen_ridge = excess_risk(beta_opt_gen_ridge, beta_star, Sigma, sigma)

        # Null risk
        risk_null = excess_risk(np.zeros(p), beta_star, Sigma, sigma)

        all_risk_opt_ridges.append(risk_opt_ridge)
        all_risk_opt_gen_ridges.append(risk_opt_gen_ridge)
        all_risk_nulls.append(risk_null)

    avg_risk_opt_ridge = np.mean(all_risk_opt_ridges)
    avg_risk_opt_gen_ridge = np.mean(all_risk_opt_gen_ridges)
    avg_risk_null = np.mean(all_risk_nulls)

    full_results[key]['avg_risk_opt_ridge'] = avg_risk_opt_ridge
    full_results[key]['avg_risk_opt_gen_ridge'] = avg_risk_opt_gen_ridge
    full_results[key]['avg_risk_null'] = avg_risk_null

    precomputed_data = []
    for seed in seeds:
        np.random.seed(seed)
        X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
        noise = np.random.randn(n) * sigma
        y = X @ beta_star + noise
        XtX = X.T @ X
        precomputed_data.append({
            'X': X, 'beta_star': beta_star, 'y': y, 'XtX': XtX
        })

    # Empirical Risks over Lambdas (using precomputed data)
    emp_R0_means = np.zeros(num_lams)
    emp_R0_sems = np.zeros(num_lams)
    emp_Rt0_means = np.zeros(num_lams)
    emp_Rt0_sems = np.zeros(num_lams)
    emp_R1_means = np.zeros(num_lams)
    emp_R1_sems = np.zeros(num_lams)
    emp_xi_means = np.zeros(num_lams)
    emp_xi_sems = np.zeros(num_lams)

    est_A_means = np.zeros(num_lams)
    est_A_sems = np.zeros(num_lams)
    est_B_means = np.zeros(num_lams)
    est_B_sems = np.zeros(num_lams)
    est_C_means = np.zeros(num_lams)
    est_C_sems = np.zeros(num_lams)
    est_xi_means = np.zeros(num_lams)
    est_xi_sems = np.zeros(num_lams)
    est_R1_means = np.zeros(num_lams)
    est_R1_sems = np.zeros(num_lams)
    est_R0_means = np.zeros(num_lams)
    est_R0_sems = np.zeros(num_lams)

    for ilam, lambda_reg in enumerate(lambda_regs):
        risk0s = []
        rtilde0s = []
        risk1s = []
        xis = []
        Atune_arr = []
        Btune_arr = []
        Ctune_arr = []
        xitune_arr = []
        R0tune_arr = []
        R1tune_arr = []

        for data in precomputed_data:
            X = data['X']
            beta_star = data['beta_star']
            y = data['y']
            XtX = data['XtX']

            # Ridge estimator and debiasing operator
            Omega = XtX / n + lambda_reg * I_p
            beta_hat_0 = np.linalg.solve(Omega, X.T @ y / n)
            M = np.linalg.solve(Omega, XtX / n)

            tilde_beta_0 = M @ beta_hat_0
            e_0 = beta_hat_0 - beta_star
            e_tilde_0 = tilde_beta_0 - beta_star

            # ABC terms
            A = e_0.T @ Sigma @ e_0
            B = e_tilde_0.T @ Sigma @ e_tilde_0
            C = e_0.T @ Sigma @ e_tilde_0

            # Risks
            xi = (A - C)/(A + B - 2*C)
            R_0 = A + sigma**2
            R_tilde_0 = B + sigma**2
            denom = A + B - 2 * C
            R_1 = ((A * B - C**2) / denom + sigma**2) if abs(denom) > 1e-12 else R_0

            risk0s.append(R_0)
            rtilde0s.append(R_tilde_0)
            risk1s.append(R_1)
            xis.append(xi)

            # Tuning A
            df_beta_hat = np.trace(M)/n
            Atune = (1/n) * np.linalg.norm(y - X @ beta_hat_0)**2 / (1 - df_beta_hat)**2 - sigma**2
            Atune_arr.append(Atune)

            # Tuning B
            df_beta_tilde = np.trace(M @ M)/n
            Btune = (1/n) * np.linalg.norm(y - X @ tilde_beta_0)**2 / (1 - df_beta_tilde)**2 - sigma**2
            Btune_arr.append(Btune)

            # Tuning C
            Ctune = ((y - X @ beta_hat_0) @ (y - X @ tilde_beta_0))/((1 - df_beta_hat) * (1 - df_beta_tilde)) / n - sigma**2
            Ctune_arr.append(Ctune)

            xitune = (Atune - Ctune)/(Atune + Btune - 2*Ctune)
            R0tune = Atune + sigma**2
            R1tune = (Atune * Btune - Ctune**2)/(Atune + Btune - 2*Ctune) + sigma**2

            xitune_arr.append(xitune)
            R0tune_arr.append(R0tune)
            R1tune_arr.append(R1tune)

        # Compute means and sems
        risk0s = np.array(risk0s)
        emp_R0_means[ilam] = np.mean(risk0s)
        emp_R0_sems[ilam] = np.std(risk0s) / np.sqrt(num_seeds)

        rtilde0s = np.array(rtilde0s)
        emp_Rt0_means[ilam] = np.mean(rtilde0s)
        emp_Rt0_sems[ilam] = np.std(rtilde0s) / np.sqrt(num_seeds)

        risk1s = np.array(risk1s)
        emp_R1_means[ilam] = np.mean(risk1s)
        emp_R1_sems[ilam] = np.std(risk1s) / np.sqrt(num_seeds)

        xis = np.array(xis)
        emp_xi_means[ilam] = np.mean(xis)
        emp_xi_sems[ilam] = np.std(xis) / np.sqrt(num_seeds)

        Atune_arr = np.array(Atune_arr)
        est_A_means[ilam] = np.mean(Atune_arr)
        est_A_sems[ilam] = np.std(Atune_arr) / np.sqrt(num_seeds)

        Btune_arr = np.array(Btune_arr)
        est_B_means[ilam] = np.mean(Btune_arr)
        est_B_sems[ilam] = np.std(Btune_arr) / np.sqrt(num_seeds)

        Ctune_arr = np.array(Ctune_arr)
        est_C_means[ilam] = np.mean(Ctune_arr)
        est_C_sems[ilam] = np.std(Ctune_arr) / np.sqrt(num_seeds)

        xitune_arr = np.array(xitune_arr)
        est_xi_means[ilam] = np.mean(xitune_arr)
        est_xi_sems[ilam] = np.std(xitune_arr) / np.sqrt(num_seeds)

        R0tune_arr = np.array(R0tune_arr)
        est_R0_means[ilam] = np.mean(R0tune_arr)
        est_R0_sems[ilam] = np.std(R0tune_arr) / np.sqrt(num_seeds)

        R1tune_arr = np.array(R1tune_arr)
        est_R1_means[ilam] = np.mean(R1tune_arr)
        est_R1_sems[ilam] = np.std(R1tune_arr) / np.sqrt(num_seeds)

    # Store all empirical and estimated results
    full_results[key]['emp_R0_means'] = emp_R0_means
    full_results[key]['emp_R0_sems'] = emp_R0_sems
    full_results[key]['emp_Rt0_means'] = emp_Rt0_means
    full_results[key]['emp_Rt0_sems'] = emp_Rt0_sems
    full_results[key]['emp_R1_means'] = emp_R1_means
    full_results[key]['emp_R1_sems'] = emp_R1_sems
    full_results[key]['emp_xi_means'] = emp_xi_means
    full_results[key]['emp_xi_sems'] = emp_xi_sems
    full_results[key]['est_A_means'] = est_A_means
    full_results[key]['est_A_sems'] = est_A_sems
    full_results[key]['est_B_means'] = est_B_means
    full_results[key]['est_B_sems'] = est_B_sems
    full_results[key]['est_C_means'] = est_C_means
    full_results[key]['est_C_sems'] = est_C_sems
    full_results[key]['est_xi_means'] = est_xi_means
    full_results[key]['est_xi_sems'] = est_xi_sems
    full_results[key]['est_R1_means'] = est_R1_means
    full_results[key]['est_R1_sems'] = est_R1_sems
    full_results[key]['est_R0_means'] = est_R0_means
    full_results[key]['est_R0_sems'] = est_R0_sems

#### Combined Plotting for R_sd and xi

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=False)

cmap = cm.get_cmap('RdPu')
colors = [cmap(0.5 + 0.4 * i / (num_pairs - 1)) for i in range(num_pairs)] if num_pairs > 1 else [cmap(0.4)]

ftsize = 14

ax_left = axes[0]
for i, pair in enumerate(pairs):
    alpha, sigma = pair
    snr = alpha**2 / sigma**2
    d = full_results[pair]
    color = colors[i]

    # Emp R1
    lower_R1 = d['emp_R1_means'] - 1 * d['emp_R1_sems']
    upper_R1 = d['emp_R1_means'] + 1 * d['emp_R1_sems']
    ax_left.fill_between(lambda_regs, lower_R1, upper_R1, alpha=0.2, color=color)
    ax_left.plot(lambda_regs, d['emp_R1_means'], '-', label=f'Emp, SNR={snr:.1f}' if i == 0 else None,
                 color=color, linewidth=2, alpha=0.7)

    # Est R1
    ax_left.plot(lambda_regs, d['est_R1_means'], 'o--', label=f'Est, SNR={snr:.1f}' if i == 0 else None,
                 color=color, linewidth=2, alpha=0.7,  markevery = 3)

    # Asym R1
    ax_left.plot(lambda_regs, d['asym_R1'], '^--', label=f'Asym, SNR={snr:.1f}' if i == 0 else None,
                 color=color, linewidth=2,  markevery = 3)

ax_left.set_xscale('log')
ax_left.set_xlabel(r'Ridge penalty $\lambda$', fontsize=ftsize + 4)
ax_left.set_ylabel('Squared prediction risk', fontsize=ftsize + 4)
ax_left.set_title(r'Self-distillation risk $R_{sd}^{\star}$', fontsize=ftsize + 4)
ax_left.grid(True, alpha=0.3)
ax_left.tick_params(axis='x', labelsize=ftsize)
ax_left.tick_params(axis='y', labelsize=ftsize)


ax_right = axes[1]
for i, pair in enumerate(pairs):
    alpha, sigma = pair
    d = full_results[pair]
    color = colors[i]

    # Emp xi
    lower_xi = d['emp_xi_means'] - 1 * d['emp_xi_sems']
    upper_xi = d['emp_xi_means'] + 1 * d['emp_xi_sems']
    ax_right.fill_between(lambda_regs, lower_xi, upper_xi, alpha=0.2, color=color)
    ax_right.plot(lambda_regs, d['emp_xi_means'], '^-', color=color, linewidth=2, alpha=0.7)

    # Est xi
    ax_right.plot(lambda_regs, d['est_xi_means'], 'o--', color=color, linewidth=2, alpha=0.7)

    # Asym xi
    ax_right.plot(lambda_regs, d['asym_xi'], '-', color=color, linewidth=2)

ax_right.set_xscale('log')
ax_right.set_xlabel(r'Ridge penalty $\lambda$', fontsize=ftsize + 4)
ax_right.set_ylabel(r'$\xi$', fontsize=ftsize + 4)
ax_right.set_title(r'Optimal mixing parameter $\xi^{\star}$', fontsize=ftsize + 4)
ax_right.grid(True, alpha=0.3)
ax_right.tick_params(axis='x', labelsize=ftsize)
ax_right.tick_params(axis='y', labelsize=ftsize)

# Collect handles and labels from left ax for shared legend
handles, labels = ax_left.get_legend_handles_labels()

ax_left.legend_ = None
ax_right.legend_ = None

snr_handles = []
snr_labels = []
for i, pair in enumerate(pairs):
    alpha, sigma = pair
    snr = alpha**2 / sigma**2
    snr_handles.append(Line2D([0], [0], color=colors[i], lw=2, label=f'SNR={snr:.1f}'))
    snr_labels.append(f'SNR={snr:.1f}')

type_handles = [
    Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Empirical'),
    Line2D([0], [0], color='gray', lw=2, marker='o', markersize=6, linestyle='--', label='Estimated', markevery = 3),
    Line2D([0], [0], color='gray', lw=2, marker='^', markersize=6, linestyle='--', label='Theoretical', markevery = 3)
]

ax_left.set_xticks([])
ax_left.set_xticks([], minor=True)
ticks = [0.01, 0.1, 1, 10, 50]
ax_left.set_xticks(ticks)
ax_left.set_xticklabels(['0.01', '0.1', '1', '10', '50'])


ax_left.set_yticks([])
ax_left.set_yticks([], minor=True)
ticks_y = [1, 1.5, 2, 2.5]
ax_left.set_yticks(ticks_y)
ax_left.set_yticklabels(['1', '1.5', '2', '2.5'])

all_handles = type_handles + snr_handles
all_labels = [h.get_label() for h in type_handles] + snr_labels

# Place shared legend
fig.legend(handles=all_handles, labels=all_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
           ncol=3, frameon=False, fontsize=ftsize + 2)
fig.tight_layout()
fig.subplots_adjust(bottom=0.05, wspace=0.4)
plt.show()