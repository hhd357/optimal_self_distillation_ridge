#### PLOT RISK CURVES OVER LAMBDA

import numpy as np
import matplotlib.pyplot as plt

# USER-DEFINED VALUES
n = 400
p = 200
alpha = 1.0
sigma = 1.0

# Covariance type and signal type
sigma_type      = "ar1_rho0.25"        # "ar1_rho0.5", "spiked", "identity", "ar1_rho0.25"
sigma_beta_type = "top_aligned"        # "top50", "bottom50", "mixed50", "identity", "top_aligned", "bottom_aligned"

spike_frac = 0.1
align_frac = 0.9

# lambda range
lambda_regs = np.logspace(np.log10(0.001), np.log10(200), 50)
num_lams    = len(lambda_regs)

np.random.seed(2025)

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

def generate_covariances(p, sigma_type, sigma_beta_type, spike_frac=0.1, align_frac=0.9):
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

    eigvals, V = np.linalg.eigh(Sigma)

    k_top = 50
    k_bottom = 50
    m_amp = 50
    amp_factor = 10.0

    if sigma_beta_type == "top50":
        D = np.ones(p) * 0.05
        D[-k_top:] = 1.0
        Sigma_beta = V @ np.diag(D) @ V.T
        trace = np.trace(Sigma_beta)
        Sigma_beta = (p / trace) * Sigma_beta
    elif sigma_beta_type == "bottom50":
        D = np.ones(p) * 0.05
        D[:k_bottom] = 1.0
        Sigma_beta = V @ np.diag(D) @ V.T
        trace = np.trace(Sigma_beta)
        Sigma_beta = (p / trace) * Sigma_beta
    elif sigma_beta_type == "mixed50":
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
        D[-k:] = align_frac * p / k
        D[:-k] = (1 - align_frac) * p / (p - k)
        Sigma_beta = V @ np.diag(D) @ V.T
    elif sigma_beta_type == "bottom_aligned":
        k = int(spike_frac * p)
        D = np.zeros(p)
        D[:k] = align_frac * p / k
        D[k:] = (1 - align_frac) * p / (p - k)
        Sigma_beta = V @ np.diag(D) @ V.T
    else:
        raise ValueError(f"Invalid sigma_beta_type: {sigma_beta_type}")

    return Sigma, Sigma_beta

def solve_kappa_bisect(lam: float, gamma: float, s: np.ndarray,
                       tol: float = 1e-12, maxit: int = 200) -> float:
    s = np.asarray(s)

    def g(k: float) -> float:
        m1 = np.mean(s/(s+k))
        return k*(1 - gamma*m1) - lam

    lo = 0.0
    hi = max(lam, 1.0)
    gh = g(hi)

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

    # Precompute proj = V.T @ beta_star
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

    # Compute asymptotic risks
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


### Main loop
Sigma, Sigma_beta = generate_covariances(p, sigma_type, sigma_beta_type,
                                         spike_frac=spike_frac, align_frac=align_frac)

gamma = p / n
I_p = np.eye(p)

cov_beta = (alpha**2 / p) * Sigma_beta
beta_star = np.random.multivariate_normal(np.zeros(p), cov_beta)

# Asymptotics
results = compute_asym_risks(gamma, Sigma, beta_star, lambda_regs, sigma)
asym_R0 = results['asym_R0']
asym_Rt0 = results['asym_Rt0']
asym_R1 = results['asym_R1']
asym_A = results['asym_A']
asym_B = results['asym_B']
asym_C = results['asym_C']

# Compute asym_xi
D = asym_A + asym_B - 2 * asym_C
mask = D > 1e-14
asym_xi = np.full_like(lambda_regs, np.nan)
asym_xi[mask] = (asym_A[mask] - asym_C[mask]) / D[mask]

# Compute sign_change_lambdas (where sign of xi changes)
valid_mask = ~np.isnan(asym_xi)
if np.sum(valid_mask) > 1:
    valid_xi = asym_xi[valid_mask]
    valid_lams = lambda_regs[valid_mask]
    sign_diff = np.diff(np.sign(valid_xi))
    changes = np.where(sign_diff != 0)[0]
    sign_change_lambdas = valid_lams[changes + 1]
else:
    sign_change_lambdas = []

### Plot Asymptotic risks
plt.style.use('default')

fig, ax = plt.subplots(1, 1, figsize=(9, 7))          # ← changed to same size

colors_main = ['tab:blue', '#A0CBE8', 'tab:green']

lw = 3
ftsize = 18

# Test risks
test_tilde_line, = ax.semilogx(lambda_regs, asym_Rt0, label=r'$\mathcal{R}_{\text{pd}}$', color=colors_main[1], linewidth=lw)
test_emp_line,   = ax.semilogx(lambda_regs, asym_R1,  label=r'$\mathcal{R}_{\text{sd}}^{\star}$', color=colors_main[2], linewidth=lw)
test_beta0_line, = ax.semilogx(lambda_regs, asym_R0,  label=r'$\mathcal{R}$', color=colors_main[0], linewidth=lw)

ax.set_xlabel(r'Ridge penalty $\lambda$', fontsize=ftsize + 4)
ax.set_ylabel('Squared prediction risk', fontsize=ftsize + 4, color='tab:blue')
ax.set_yscale('log')

ax.set_yticks([])
ax.set_yticks([], minor=True)
ticks = [1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
ax.set_yticks(ticks)
ax.set_yticklabels(['1', '1.2', '1.4', '1.6', '1.8', '2.0', '2.2'], fontsize=ftsize)

ax.tick_params(axis='y', labelsize=ftsize, labelcolor='tab:blue')
ax.tick_params(axis='x', labelsize=ftsize)
ax.spines['left'].set_color('tab:blue')
ax.grid(True, alpha=0.3)

ax.set_title(f'Asymptotic risks', fontsize=ftsize + 4)

# Secondary y-axis for xi
ax_twin = ax.twinx()

# Find lambda where asym_xi is closest to 0
asym_xi_np = np.array(asym_xi)
valid_idx = np.where(~np.isnan(asym_xi_np))[0]
if len(valid_idx) > 0:
    idx_zero = valid_idx[np.argmin(np.abs(asym_xi_np[valid_idx]))]
    lambda_zero = lambda_regs[idx_zero]

    idx_one = valid_idx[np.argmin(np.abs(asym_xi_np[valid_idx] - 1))]
    lambda_one = lambda_regs[idx_one]
else:
    lambda_zero = lambda_regs[0]
    lambda_one = lambda_regs[-1]
    idx_zero = 0
    idx_one = -1

test_xiemp_line, = ax_twin.semilogx(lambda_regs, asym_xi, label=r'$\xi$', color='tab:red', linestyle='-', linewidth=lw)
ax_twin.set_ylabel(r'Optimal mixing parameter', fontsize=ftsize + 4, color='tab:red')
ax_twin.tick_params(axis='y', labelsize=ftsize, labelcolor='tab:red')
ax_twin.spines['right'].set_color('tab:red')

ax_twin.set_xticks([])
ax_twin.set_xticks([], minor=True)
ticks = [1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 5, 50, 200]
ax_twin.set_xticks(ticks)
ax_twin.set_xticklabels(['1e-5', '1e-4','0.001','0.01', '0.1', '1', '5', '50', '200'], fontsize=ftsize)

# vertical lines, stars, annotations
for lam in sign_change_lambdas:
    ax.axvline(x=lam, color='tab:gray', linestyle='--', alpha=0.7, linewidth=2)

ax.axvline(x=lambda_one, color='tab:gray', linestyle='--', alpha=0.7, linewidth=2)

idx_min_R0 = np.argmin(asym_R0)
ax.plot(lambda_regs[idx_min_R0], asym_R0[idx_min_R0], marker='*', color=colors_main[0], markersize=24, markeredgecolor='white', markeredgewidth=1, zorder=5)

idx_min_Rtilde = np.argmin(asym_Rt0)
ax.plot(lambda_regs[idx_min_Rtilde], asym_Rt0[idx_min_Rtilde], marker='*', color=colors_main[1], markersize=24, markeredgecolor='white', markeredgewidth=1, zorder=5)

idx_min_R1 = np.nanargmin(asym_R1)
ax.plot(lambda_regs[idx_min_R1], asym_R1[idx_min_R1], marker='*', color=colors_main[2], markersize=24, markeredgecolor='white', markeredgewidth=1, zorder=5)

ax_twin.annotate(r'$\xi^{\star} = 0$', xy=(lambda_zero, asym_xi[idx_zero]),
                 xytext=(lambda_zero * 1.2, asym_xi[idx_zero] * 8),
                 color='tab:red', fontsize=ftsize, ha='left')

ax_twin.annotate(r'$\xi^{\star} = 1$', xy=(lambda_one, asym_xi[idx_one]),
                 xytext=(lambda_one * 0.28, asym_xi[idx_one] * 5),
                 color='tab:red', fontsize=ftsize, ha='left')

y_low, y_high = ax.get_ylim()
y_text = y_low + 0.93 * (y_high - y_low)

xi1_line = lambda_one
xi0_line = lambda_zero
left_mid  = np.exp((0.07 * np.log(lambda_regs[0]) + 0.85 * np.log(xi1_line)))
right_mid = np.exp((0.95 * np.log(xi0_line) + 0.05 * np.log(lambda_regs[-1])))

ax.text(left_mid, y_text, r'anti-learning $\xi^{\star} > 1$', ha='right', va='bottom', fontsize=ftsize - 2, color='black')
ax.text(right_mid, y_text, r'pro-learning $\xi^{\star} < 0$', ha='left', va='bottom', fontsize=ftsize - 2, color='black')

# Legend
all_test_lines = [test_beta0_line, test_tilde_line, test_emp_line, test_xiemp_line]
all_test_labels = [l.get_label() for l in all_test_lines]
fig.legend(all_test_lines, all_test_labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=4, fontsize=ftsize + 2)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()

### Plot Normalized gain

fig, ax = plt.subplots(1, 1, figsize=(9, 7))

lw = 3
ftsize = 18

sigma2 = sigma**2
num = asym_R0 - asym_R1
denom = asym_R0 - sigma2
gain = np.full_like(lambda_regs, np.nan, dtype=float)
mask_gain = np.abs(denom) > 1e-8
gain[mask_gain] = num[mask_gain] / denom[mask_gain]

gain_line, = ax.semilogx(lambda_regs, gain, label=r'Normalized gain $\frac{\mathcal{R} - \mathcal{R}_{\text{sd}}^{\star} }{\mathcal{R} - \sigma^{2}}$', color='tab:purple', linewidth=lw)

ax.set_title(f'Normalized gain of SD risk', fontsize=ftsize + 4)
ax.set_xlabel(r'Ridge penalty $\lambda$', fontsize=ftsize + 4)
ax.set_ylabel('Normalized gain', fontsize=ftsize + 4, color='tab:purple')
ax.tick_params(axis='y', labelsize=ftsize, labelcolor='tab:purple')
ax.tick_params(axis='x', labelsize=ftsize)
ax.spines['left'].set_color('tab:purple')
ax.grid(True, alpha=0.3)

ax.set_xticks([])
ax.set_xticks([], minor=True)
ticks = [0.001, 0.01, 0.1, 1, 5, 50, 200]
ax.set_xticks(ticks)
ax.set_xticklabels(['0.001','0.01', '0.1', '1', '5', '50', '200'], fontsize=ftsize)

# Legend
fig.legend([gain_line], [gain_line.get_label()], loc='lower center', bbox_to_anchor=(0.5, -0.03), fontsize=ftsize + 2)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()