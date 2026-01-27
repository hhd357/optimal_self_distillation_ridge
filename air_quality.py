#INSTALL DATASETS

from ucimlrepo import fetch_ucirepo

import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# fetch dataset
air_quality = fetch_ucirepo(id=360)

# AIR QUALITY DATASET

import pandas as pd
import numpy as np

df = air_quality.data.original

# Define X covariates columns
x_columns = [
    'PT08.S1(CO)',
    'PT08.S2(NMHC)',
    'PT08.S3(NOx)',
    'PT08.S4(NO2)',
    'PT08.S5(O3)',
    'T',
    'RH',
    'AH'
]

# Define y target column
y_column = 'NO2(GT)'

# Extract X and y
X = df[x_columns]
y = df[y_column]

# Exclude records where any value in X or y is negative
mask = (X >= -100).all(axis=1) & (y >= 0)
X = X[mask]
y = y[mask]

np.random.seed(2026)

# Sequential split 70-30
n = len(X)
indices = np.arange(n)
split = int(0.7*n)
train_idx = indices[:split]
test_idx = indices[split:]

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_mask = X_train.notna().all(axis=1) & y_train.notna()
X_train = X_train[train_mask]
y_train = y_train[train_mask]

test_mask = X_test.notna().all(axis=1) & y_test.notna()
X_test = X_test[test_mask]
y_test = y_test[test_mask]


# Coordinate-wise whitening (standardization) for all X features and Y target
# Compute means and stds on the train set only

X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
y_train_mean = y_train.mean()
y_train_std = y_train.std()

# Apply to train
X_train = (X_train - X_train_mean) / X_train_std
y_train = (y_train - y_train_mean) / y_train_std

# Apply to test using train stats
X_test = (X_test - X_train_mean) / X_train_std
y_test = (y_test - y_train_mean) / y_train_std

n_train = len(X_train)
n_test = len(X_test)

p = X_train.shape[1]
print("no of covariates", p)

# Print the number of valid records and split sizes after transformations
print(f"Number of valid records before split: {len(X)}")
print(f"n_train: {n_train}")
print(f"n_test: {n_test}")

### Main code to run
p = X_train.shape[1]
n = n_train

I_p = np.eye(p)

# Test error measured on test set
def test_mse(beta):
    return np.mean((y_test - X_test @ beta)**2)

# Train error measured on train set
def train_mse(beta):
    return np.mean((y_train - X_train @ beta)**2)

# Initialize arrays
Aemp_arr = []
Bemp_arr = []
Cemp_arr = []
xi_emp_arr = []
Atune_arr = []
Btune_arr = []
Ctune_arr = []
xitune_arr = []
R0_arr = []
Rtilde_arr = []
R1_arr = []
R1_tune_arr = []
train_R0_arr = []
train_R1_arr = []
train_Rtilde_arr = []
train_R1_tune_arr = []

# Set the lambda range
lambda_regs = np.logspace(np.log10(1e-3), np.log10(50), 100)

num_lams = len(lambda_regs)

XtX = X_train.T @ X_train
I_p = np.eye(p)

for ilam, lambda_reg in enumerate(lambda_regs):
    # Ridge estimator and debiasing operator
    Omega = XtX / n + lambda_reg * I_p
    beta_hat_0 = np.linalg.solve(Omega, X_train.T @ y_train / n)
    M = np.linalg.solve(Omega, XtX / n)
    tilde_beta_0 = M @ beta_hat_0

    # Empirical
    A_emp = test_mse(beta_hat_0)
    Aemp_arr.append(A_emp)

    B_emp = test_mse(tilde_beta_0)
    Bemp_arr.append(B_emp)

    C_emp = ((y_test - X_test @ beta_hat_0).T @ (y_test - X_test @ tilde_beta_0)) / n_test
    Cemp_arr.append(C_emp)

    xi_emp = (A_emp - C_emp)/(A_emp + B_emp - 2*C_emp)
    xi_emp_arr.append(xi_emp)

    beta_hat_1 = xi_emp * tilde_beta_0 + (1 - xi_emp) * beta_hat_0
    R_1 = test_mse(beta_hat_1)

    # Tuning A
    df_beta_hat = np.trace(M)/n
    Atune = (1/n) * np.linalg.norm(y_train - X_train @ beta_hat_0)**2 / (1 - df_beta_hat)**2
    Atune_arr.append(Atune)

    # Tuning B
    df_beta_tilde = np.trace(M @ M)/n
    Btune = (1/n) * np.linalg.norm(y_train - X_train @ tilde_beta_0)**2 / (1 - df_beta_tilde)**2
    Btune_arr.append(Btune)

    # Tuning C
    Ctune = ((y_train - X_train @ beta_hat_0) @ (y_train - X_train @ tilde_beta_0)) / ((1 - df_beta_hat) * (1 - df_beta_tilde)) / n
    Ctune_arr.append(Ctune)

    # Tuning xi
    xitune = (Atune - Ctune)/(Atune + Btune - 2*Ctune)
    xitune_arr.append(xitune)

    beta_hat_1_tune = xitune * tilde_beta_0 + (1 - xitune) * beta_hat_0

    R_0 = test_mse(beta_hat_0)
    R_tilde_0 = test_mse(tilde_beta_0)
    R_1_tune = test_mse(beta_hat_1_tune)

    train_R_0 = train_mse(beta_hat_0)
    train_R_1 = train_mse(beta_hat_1)
    train_R_tilde_0 = train_mse(tilde_beta_0)
    train_R_1_tune = train_mse(beta_hat_1_tune)

    # save over lambdas
    R0_arr.append(R_0)
    R1_arr.append(R_1)
    Rtilde_arr.append(R_tilde_0)
    R1_tune_arr.append(R_1_tune)
    train_R0_arr.append(train_R_0)
    train_R1_arr.append(train_R_1)
    train_Rtilde_arr.append(train_R_tilde_0)
    train_R1_tune_arr.append(train_R_1_tune)

# Detect sign changes in xi_emp_arr
sign_change_lambdas = []
for i in range(1, len(xi_emp_arr)):
    if xi_emp_arr[i-1] * xi_emp_arr[i] < 0:
        # Interpolate to find approximate zero-crossing lambda for smoother vertical line
        lam1, lam2 = lambda_regs[i-1], lambda_regs[i]
        xi1, xi2 = xi_emp_arr[i-1], xi_emp_arr[i]
        zero_lam = lam1 * np.exp(np.log(lam2 / lam1) * (-xi1 / (xi2 - xi1)))
        sign_change_lambdas.append(zero_lam)
    elif xi_emp_arr[i] == 0:
        sign_change_lambdas.append(lambda_regs[i])

### Main code to plot
plt.style.use('default')

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

colors_main = ['tab:blue', '#A0CBE8', 'tab:green']

lw = 3
ftsize = 18

# Test risks curves
test_tilde_line, = ax.semilogx(lambda_regs, Rtilde_arr, label=r'$R_{\text{pd}}$', color=colors_main[1], linewidth=lw)
test_emp_line, = ax.semilogx(lambda_regs, R1_arr, label=r'$R_{\text{sd}}^{\star}$', color=colors_main[2], linewidth=lw)
test_est_line, = ax.semilogx(lambda_regs, R1_tune_arr, label=r'$\widehat{R}_{\text{sd}}$', color=colors_main[2], \
                             linestyle='--', marker='o', markersize=6, linewidth=lw, alpha=0.5, markevery=7)
test_beta0_line, = ax.semilogx(lambda_regs, R0_arr, label=r'$R$', color=colors_main[0], linewidth=lw)
ax.set_xlabel(r'Ridge penalty $\lambda$', fontsize=ftsize + 4)
ax.set_ylabel('Squared prediction risk', fontsize=ftsize + 4, color='tab:blue')
ax.set_yscale('log')

ax.set_yticks([])
ax.set_yticks([], minor=True)
ticks = [0.4, 0.6, 0.8, 1, 1.2]
ax.set_yticks(ticks)
ax.set_yticklabels(['0.4', '0.6', '0.8', '1', '1.2'], fontsize=ftsize)

ax.tick_params(axis='y', labelsize=ftsize, labelcolor='tab:blue')
ax.tick_params(axis='x', labelsize=ftsize)  # Added this line for x-axis tick labels
ax.spines['left'].set_color('tab:blue')
ax.grid(True, alpha=0.3)

ax.set_title('Air Quality', fontsize=ftsize + 4)

# Secondary y-axis for xi
ax_twin = ax.twinx()

# Find lambda where xi_emp_arr is closest to 0
xi_emp_np = np.array(xi_emp_arr)
idx_zero = np.argmin(np.abs(xi_emp_np))
lambda_zero = lambda_regs[idx_zero]

#
test_xiemp_line, = ax_twin.semilogx(lambda_regs, xi_emp_arr, label=r'$\xi^{*}$', color='tab:red', linestyle='-', linewidth=lw)
ax_twin.set_ylabel(r'Optimal mixing parameter', fontsize=ftsize + 4, color='tab:red')
ax_twin.tick_params(axis='y', labelsize=ftsize, labelcolor='tab:red')
ax_twin.spines['right'].set_color('tab:red')

ax_twin.set_xticks([])
ax_twin.set_xticks([], minor=True)
ticks = [1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 5, 50, 200]
ax_twin.set_xticks(ticks)
ax_twin.set_xticklabels(['1e-5', '1e-4','0.001','0.01', '0.1', '1', '5', '50', '200'], fontsize=ftsize)

# Add vertical lines for sign changes
for lam in sign_change_lambdas:
    ax.axvline(x=lam, color='tab:gray', linestyle='--', alpha=0.7, linewidth=2)

# Find lambda where xi_emp_arr is closest to 1 and add vertical line
idx_one = np.argmin(np.abs(xi_emp_np - 1))
lambda_one = lambda_regs[idx_one]
ax.axvline(x=lambda_one, color='tab:gray', linestyle='--', alpha=0.7, linewidth=2)

# Find minimum of R0 and add star
idx_min_R0 = np.argmin(R0_arr)
lambda_min_R0 = lambda_regs[idx_min_R0]
R0_min = R0_arr[idx_min_R0]
ax.plot(lambda_min_R0, R0_min, marker='*', color=colors_main[0], markersize=24, markeredgecolor='white', markeredgewidth=1, zorder=5)

# Find minimum of R0 and add star
idx_min_Rtilde = np.argmin(Rtilde_arr)
lambda_min_Rtilde = lambda_regs[idx_min_Rtilde]
Rtilde_min = Rtilde_arr[idx_min_Rtilde]
ax.plot(lambda_min_Rtilde, Rtilde_min, marker='*', color=colors_main[1], markersize=24, markeredgecolor='white', markeredgewidth=1, zorder=5)

# Find minimum of R1 and add star
idx_min_R1 = np.argmin(R1_arr)
lambda_min_R1 = lambda_regs[idx_min_R1]
R1_min = R1_arr[idx_min_R1]
ax.plot(lambda_min_R1, R1_min, marker='*', color=colors_main[2], markersize=24, markeredgecolor='white', markeredgewidth=1, zorder=5)

# Annotate the red star with a red arrow and red text
ax_twin.annotate(r'$\xi^{*} = 0$', xy=(lambda_zero, xi_emp_np[idx_zero]),
                 xytext=(lambda_zero * 1.2, xi_emp_np[idx_zero] * 1.5),
                 color='tab:red', fontsize=ftsize, ha='left')

# Annotate the red star at xi=1 with a red arrow and red text
ax_twin.annotate(r'$\xi^{*} = 1$', xy=(lambda_one, xi_emp_np[idx_one]),
                 xytext=(lambda_one * 0.28, xi_emp_np[idx_one] * 0.2),
                 color='tab:red', fontsize=ftsize, ha='left')


# Region annotations
y_low, y_high = ax.get_ylim()
y_text = y_low + 0.9 * (y_high - y_low)

xi1_line = lambda_one
xi0_line = lambda_zero
# Assuming xi1_line < xi0_line
left_mid = np.exp((0.07* np.log(lambda_regs[0]) + 0.85 * np.log(xi1_line)))
right_mid = np.exp((0.95*np.log(xi0_line) + 0.05*np.log(lambda_regs[-1])))

ax.text(left_mid, y_text, r'anti-learning $\xi^{*} > 1$', ha='right', va='bottom', fontsize=ftsize - 2, color='black')

ax.text(right_mid, y_text, r'pro-learning $\xi^{*} < 0$', ha='left', va='bottom', fontsize= ftsize - 2, color='black')

all_test_lines = [test_beta0_line, test_tilde_line, test_emp_line, test_est_line, test_xiemp_line]
all_test_labels = [l.get_label() for l in all_test_lines]
ax.legend(all_test_lines, all_test_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=ftsize + 2)

plt.tight_layout()
plt.show()