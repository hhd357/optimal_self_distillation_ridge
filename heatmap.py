# HEATMAP FOR EXTREME REGULARIZED RISKS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as ticker

# USER-DEFINED VALUES
n = 200
sigma = 1.0

# Gamma and SNR grids
gammas = np.logspace(np.log10(0.1), np.log10(10), 50)
snrs = np.logspace(np.log10(0.1), np.log10(10), 50)
alphas = np.sqrt(snrs) * sigma  # alpha = sqrt(SNR) * sigma

# Covariance type selections
sigma_type = "identity"        # Options: "ar1_rho0.5", "spiked", "identity", "equicorrelated", "ar1_rho0.25"
sigma_beta_type = "identity"   # Options: "top50", "bottom50", "mixed50", "identity"

# Fixed lambdas for evaluation
lambda_large = 1e6
lambda_small = 1e-3

# Covariance Generation

def ar1_covariance(p, rho):
    i = np.arange(p)
    j = np.arange(p)
    return rho ** np.abs(i[:, None] - j[None, :])

def spiked_covariance(p, num_spikes=1, spike_strength=5.0):
    Sigma = np.eye(p)
    for _ in range(num_spikes):
        v = np.random.randn(p)
        v /= np.linalg.norm(v)
        Sigma += spike_strength * np.outer(v, v)
    return Sigma

def equicorrelated_covariance(p, rho=0.5):
    Sigma = np.full((p, p), rho)
    np.fill_diagonal(Sigma, 1.0)
    return Sigma

def toeplitz_exp(n, alpha):
    idx = np.arange(n)
    return np.exp(-alpha * np.abs(idx[:, None] - idx[None, :]))

def random_wishart(p, m):
    X = np.random.randn(m, p)
    return X.T @ X / m

def block_cov(block_sizes, rho_within=0.8, rho_between=0.0):
    n_total = sum(block_sizes)
    Sigma = np.full((n_total, n_total), rho_between)
    start = 0
    for b in block_sizes:
        Sigma[start:start+b, start:start+b] = rho_within
        start += b
    # make diagonals 1
    np.fill_diagonal(Sigma, 1.0)
    return Sigma

# Fixed-Point Solver and Risk Computation
def solve_v_fixed_point(lambda_, gamma, eigs, tol=1e-15, maxit=200, v_init=1):
    v = float(v_init)

    def a1(v):
        return float(np.mean(eigs / (v * eigs + 1.0)))

    def a2(v):
        denom = v * eigs + 1.0
        return float(np.mean((eigs ** 2) / (denom ** 2)))

    def a3(v):
        denom = v * eigs + 1.0
        return float(np.mean((eigs ** 3) / (denom ** 3)))

    def a4(v):
        denom = v * eigs + 1.0
        return float(np.mean((eigs ** 4) / (denom ** 4)))

    for i in range(maxit):
        k = i
        f = 1.0 / v - lambda_ - gamma * a1(v)
        fprime = -1.0 / (v ** 2) + gamma * a2(v)
        step = -f / (fprime)
        t = 1.0
        new_v = v + t * step
        while new_v <= 0.0:
            t *= 0.5
            new_v = v + t * step
            if t < 1e-12:
                new_v = max(v * 0.5, 1e-12)
                break
        if abs(new_v - v) <= tol * max(1.0, abs(v)):
            v = new_v
            break
        v = new_v
        if v < 1e-12:
          print('alert')
        v = max(v, 1e-12)
    return v, a1(v), a2(v), a3(v), a4(v)

def compute_asymptotic_risk(lambda_reg, alpha, gamma, eigs, sigma):
    """Compute R_1 asymptotic risk for given lambda, alpha, gamma, eigs, sigma."""
    r = alpha
    v, a1, a2, a3, a4 = solve_v_fixed_point(
        lambda_=lambda_reg, gamma=gamma, eigs=eigs, tol=1e-15, maxit=500, v_init=0.001
    )

    if np.isnan(v) or v <= 0:
        return np.inf

    v_tilde = (gamma * a2) / (1 / (v ** 2) - gamma * a2)

    A_v = -2 * (v ** -3) + 2 * gamma * a3
    v_prime_1 = - (v ** 2) * (1 + v_tilde)
    v_prime_2 = 2 * (v ** 3) * ((1 + v_tilde) ** 2) + 2 * gamma * (v ** 5) * ((1 + v_tilde) ** 3) * (a2 - v * a3)
    v_prime_3 = 6 * (v ** (-4) - gamma * a4) * (v_prime_1 ** 4) + 3 * (A_v ** 2) * (v_prime_1 ** 5)

    a2_prime = -2 * a3
    a2_dprime = 6 * a4

    # Asymptotic terms
    T1_asym = a1 / lambda_reg if lambda_reg > 0 else 0
    T2_asym = a1 / (lambda_reg ** 2) - (1 / lambda_reg) * (v ** 2) * (1 + v_tilde) * a2 if lambda_reg > 0 else 0
    T3_asym = a1 / (lambda_reg ** 3) + 1 / (lambda_reg ** 2) * a2 * v_prime_1 - 1 / (2 * lambda_reg) * (-2 * a3 * (v_prime_1 ** 2) + a2 * v_prime_2) if lambda_reg > 0 else 0
    T4_asym = (a1 / (lambda_reg ** 4) + (1 / (lambda_reg ** 3)) * a2 * v_prime_1
               - (1 / (2 * (lambda_reg ** 2))) * (a2_prime * (v_prime_1 ** 2) + a2 * v_prime_2)
               + (1 / (6 * lambda_reg)) * (a2_dprime * (v_prime_1 ** 3) + 3 * a2_prime * v_prime_1 * v_prime_2 + a2 * v_prime_3)) if lambda_reg > 0 else 0

    # ABC terms
    A_asym = r**2 * (lambda_reg**2 * T2_asym) + (sigma ** 2) * gamma * (T1_asym - lambda_reg * T2_asym)
    B_asym = r**2 * (4 * lambda_reg**2 * T2_asym - 4 * lambda_reg**3 * T3_asym + lambda_reg**4 * T4_asym) \
             + (sigma ** 2) * gamma * (T1_asym - 3 * lambda_reg * T2_asym + 3 * lambda_reg**2 * T3_asym - lambda_reg**3 * T4_asym)
    C_asym = r**2 * (2 * lambda_reg**2 * T2_asym - lambda_reg**3 * T3_asym) \
             + (sigma ** 2) * gamma * (T1_asym - 2 * lambda_reg * T2_asym + lambda_reg**2 * T3_asym)

    # Asymptotic risks
    R_0_asym = A_asym + (sigma ** 2)
    denom_r1 = A_asym + B_asym - 2 * C_asym
    R_1_asym = (A_asym * B_asym - C_asym ** 2) / denom_r1 + (sigma ** 2)

    return R_1_asym

def compute_optimal_ridge_risk(alpha, gamma, eigs, sigma):
    """Compute asymptotic optimal ridge risk R* = min R_0(lambda)."""
    lambda_opt = gamma * (sigma**2 / alpha**2)
    r = alpha
    v, a1, a2, a3, a4 = solve_v_fixed_point(
        lambda_=lambda_opt, gamma=gamma, eigs=eigs, tol=1e-15, maxit=500, v_init=0.001
    )
    if np.isnan(v) or v <= 0:
        return sigma ** 2  # Fallback to null risk

    v_tilde = (gamma * a2) / (1 / (v ** 2) - gamma * a2) if abs(1 / (v ** 2) - gamma * a2) > 1e-18 else 0
    T1_asym = a1 / lambda_opt if lambda_opt > 0 else 0
    T2_asym = a1 / (lambda_opt ** 2) - (1 / lambda_opt) * (v ** 2) * (1 + v_tilde) * a2 if lambda_opt > 0 else 0
    A_asym = r**2 * (lambda_opt**2 * T2_asym) + (sigma ** 2) * gamma * (T1_asym - lambda_opt * T2_asym)
    R_star = A_asym + (sigma ** 2)
    return R_star

#### Compute Ratios
# Storage for ratios
ratios_large = np.zeros((len(gammas), len(snrs)))
ratios_small = np.zeros((len(gammas), len(snrs)))

print("Computing asymptotic risks for 100 settings...")
for i, gamma in enumerate(gammas):
    p = int(round(gamma * n))  # p = gamma * n
    print(f"Processing gamma={gamma:.2f}, p={p}")

    # Generate eigs for this p (optimized for AR(1))
    if sigma_type in ["ar1_rho0.5", "ar1_rho0.25"]:
        if sigma_type == "ar1_rho0.5":
            rho = 0.5
        else:
            rho = 0.25
        theta = np.pi * np.arange(1, p+1) / (p + 1)
        eigs = (1 - rho**2) / (1 + rho**2 - 2 * rho * np.cos(theta))
    else:
        # For other types, generate full matrix
        if sigma_type == "spiked":
            Sigma_base = spiked_covariance(p)
        elif sigma_type == "identity":
            Sigma_base = np.eye(p)
        elif sigma_type == "equicorrelated":
            Sigma_base = equicorrelated_covariance(p)
        else:
            raise ValueError(f"Invalid sigma_type: {sigma_type}")
        eigvals, V = np.linalg.eigh(Sigma_base)
        eigs = eigvals

    # Generate Sigma_beta_base
    k_top = 50
    k_bottom = 50
    m_amp = 50
    amp_factor = 10.0

    if sigma_beta_type == "top50":
        D = np.ones(p) * 0.05
        if p >= k_top:
            D[-k_top:] = 1.0
        else:
            D = np.ones(p)  # Adjust if p < k_top
        Sigma_beta_base = V @ np.diag(D) @ V.T
        Sigma_beta_base *= p / np.trace(Sigma_beta_base)
    elif sigma_beta_type == "bottom50":
        D = np.ones(p) * 0.05
        if p >= k_bottom:
            D[:k_bottom] = 1.0
        else:
            D = np.ones(p)
        Sigma_beta_base = V @ np.diag(D) @ V.T
        Sigma_beta_base *= p / np.trace(Sigma_beta_base)
    elif sigma_beta_type == "mixed50":
        D = np.ones(p)
        if p >= m_amp:
            D[:m_amp] *= amp_factor
        Sigma_beta_base = np.diag(D)
        Sigma_beta_base *= p / np.trace(Sigma_beta_base)
    elif sigma_beta_type == "identity":
        Sigma_beta_base = np.eye(p)
    else:
        raise ValueError(f"Invalid sigma_beta_type: {sigma_beta_type}")

    for j, alpha in enumerate(alphas):
        # Compute R*
        R_star = compute_optimal_ridge_risk(alpha, gamma, eigs, sigma)

        # Compute R1 at large lambda
        R1_large = compute_asymptotic_risk(lambda_large, alpha, gamma, eigs, sigma)
        ratios_large[i, j] = (R1_large - R_star) / R_star if R_star > 1e-10 else 0

        # Compute R1 at small lambda
        R1_small = compute_asymptotic_risk(lambda_small, alpha, gamma, eigs, sigma)
        ratios_small[i, j] = (R1_small - R_star) / R_star if R_star > 1e-10 else 0

        if R1_small < R_star:
          print(f"For gamma={gamma:.3f}, alpha={alpha:.3f}: R_star={R_star:.6f}, R1_small={R1_small:.6f}")


### Plot Heatmaps

# Compute log ratios for color scaling
log_ratios_large = np.log10(np.maximum(ratios_large, 1e-12))
log_ratios_small = np.log10(np.maximum(ratios_small, 1e-12))

# Global range for unified color scale
global_min = min(np.min(log_ratios_large), np.min(log_ratios_small))
global_max = max(np.max(log_ratios_large), np.max(log_ratios_small))
print(f"Global range: min={global_min:.3f}, max={global_max:.3f}")

# Fix white midpoint at -2 while spanning full data range asymmetrically
vcenter = -2
norm = TwoSlopeNorm(vmin=global_min, vcenter=vcenter, vmax=global_max)
print(f"Color scale: vmin={global_min:.3f}, vcenter={vcenter}, vmax={global_max:.3f}")

ftsize = 12

# Custom formatter
def exp_formatter(x, pos):
    return r'$10^{{{:.0f}}}$'.format(x)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

im1 = axes[0].pcolormesh(snrs, gammas, log_ratios_small, shading='auto', cmap='coolwarm', norm=norm)
levels = [-4, -3, -2, -1]
cs1 = axes[0].contour(snrs, gammas, log_ratios_small, levels=levels, colors=['blue', 'tab:blue', 'tab:orange', 'tab:red'], linewidths=1.0)

label_texts = {-4: '0.01%', -3: '0.1%', -2: '1%', -1: '10%'}
for i, level in enumerate(levels):
    segs = cs1.allsegs[i]
    if len(segs) > 0:
        seg = segs[0]  # Take the first segment (assuming main curve)
        if len(seg) > 0:
            mid_idx = len(seg) // 2
            mid_x, mid_y = seg[mid_idx]
            axes[0].text(mid_x, mid_y, label_texts[level], fontsize=10, color='black', fontweight='bold',
                         ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))
axes[0].set_xlabel('SNR', fontsize=ftsize + 4)
axes[0].set_ylabel(r'$\gamma$', fontsize=ftsize + 4)
axes[0].set_title(r'$\lim_{\lambda \to 0}  \: (\mathcal{R}_{sd}^{*} - \mathcal{R}^{*})/ \mathcal{R}^{*}$', fontsize=ftsize + 4)
axes[0].set_xscale('log')
axes[0].set_yscale('log')

axes[0].set_xticks([])
axes[0].set_xticks([], minor=True)
ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
axes[0].set_xticks(ticks)
axes[0].set_xticklabels(['0.1', '0.2', '0.5', '1', '2', '5', '10'], fontsize=ftsize)

axes[0].set_yticks([])
axes[0].set_yticks([], minor=True)
axes[0].set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
axes[0].set_yticklabels(['0.1', '0.2', '0.5', '1', '2', '5', '10'], fontsize=ftsize)

cbar1 = plt.colorbar(im1, ax=axes[0], aspect=20)
global_ticks = [-8, -6, -4, -2, -1, 0, 1]
cbar1.ax.yaxis.set_major_locator(ticker.FixedLocator(global_ticks))
cbar1.ax.yaxis.set_major_formatter(ticker.FuncFormatter(exp_formatter))
cbar1.ax.tick_params(labelsize=ftsize)

im2 = axes[1].pcolormesh(snrs, gammas, log_ratios_large, shading='auto', cmap='coolwarm', norm=norm)
cs2 = axes[1].contour(snrs, gammas, log_ratios_large, levels=levels, colors=['blue', 'tab:blue', 'tab:orange', 'tab:red'], linewidths=1.0)

for i, level in enumerate(levels):
    segs = cs2.allsegs[i]
    if len(segs) > 0:
        seg = segs[0]
        if len(seg) > 0:
            mid_idx = len(seg) // 2
            mid_x, mid_y = seg[mid_idx]
            axes[1].text(mid_x, mid_y, label_texts[level], fontsize=10, color='black', fontweight='bold',
                         ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_ylabel(r'$\gamma$', fontsize=ftsize + 4)

axes[1].set_xticks([])
axes[1].set_xticks([], minor=True)
ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
axes[1].set_xticks(ticks)
axes[1].set_xticklabels(['0.1', '0.2', '0.5', '1', '2', '5', '10'], fontsize=ftsize)

axes[1].set_yticks([])
axes[1].set_yticks([], minor=True)
axes[1].set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
axes[1].set_yticklabels(['0.1', '0.2', '0.5', '1', '2', '5', '10'], fontsize=ftsize)
axes[1].set_xlabel('SNR', fontsize=ftsize + 4)
axes[1].set_title(r'$\lim_{\lambda \to \infty} \: (\mathcal{R}_{sd}^{*} - \mathcal{R}^{*})/ \mathcal{R}^{*}$', fontsize=ftsize + 4)

cbar2 = plt.colorbar(im2, ax=axes[1])
cbar2.ax.yaxis.set_major_locator(ticker.FixedLocator(global_ticks))
cbar2.ax.yaxis.set_major_formatter(ticker.FuncFormatter(exp_formatter))
cbar2.ax.tick_params(labelsize=ftsize)  # Set colorbar tick label font size

plt.tight_layout()
fig.subplots_adjust(wspace=0.3)
plt.show()