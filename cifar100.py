# ResNet-34 features pretrained on ImageNet, with CIFAR-100 dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CIFAR-100 dataset
transform_base = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

# Load full train and test sets 
train_dataset_full = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_base)
test_dataset_full = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_base)

# Load pretrained ResNet-34 and modify for feature extraction
model = torchvision.models.resnet34(pretrained=True)
model.fc = nn.Identity()  # Remove final FC layer, output 512-dim features
model = model.to(device)
model.eval()

### SUBSAMPLING CIFAR DATASETS
K = 10
n_train = 20000 #20000 for CIFAR100
n_test = 10000 #10000 for CIFAR100
p = 512

np.random.seed(2026)

# Subsample
train_indices = np.random.choice(len(train_dataset_full), n_train, replace=False)
test_indices = np.random.choice(len(test_dataset_full), n_test, replace=False)

train_dataset = Subset(train_dataset_full, train_indices)
test_dataset = Subset(test_dataset_full, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Extract features
def extract_features(loader):
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            feats = model(images)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
    return np.vstack(features), np.hstack(labels)

# Extract train and test features
print("Extracting train features...")
X_train, y_train = extract_features(train_loader) #size n_train * p
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

print("Extracting test features...")
X_test, y_test = extract_features(test_loader) #size n_test * p
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# One-hot encode y for multi-output
encoder = OneHotEncoder(sparse_output=False)
y_train_multi = encoder.fit_transform(y_train.reshape(-1, 1)) #size n_train * K
y_test_multi = encoder.transform(y_test.reshape(-1, 1))  #size n_test * K

#### Main code to run

# Lambda range
lambda_regs = np.logspace(np.log10(1e-3), np.log10(50), 100)
num_lams = len(lambda_regs)

# Test error measured on test set
def test_mse(beta_all):
    return (1/n_test) * sum(np.linalg.norm(y_test_multi[:, i] - X_test @ beta_all[i])**2 for i in range(K))

# Train error measured on train set
def train_mse(beta_all):
    return (1/n_train) * sum(np.linalg.norm(y_train_multi[:, i] - X_train @ beta_all[i])**2 for i in range(K))

xi_emp_arr = []
xitune_arr = []
R0_arr = []
Rtilde_arr = []
R1_arr = []
R1_tune_arr = []
train_R0_arr = []
train_R1_arr = []
train_Rtilde_arr = []
train_R1_tune_arr = []
Aemp_arr = []
Bemp_arr = []
Cemp_arr = []

# Arrays for test accuracies over lambdas
acc0_arr = []
acc_tilde_arr = []
acc1_arr = []

def classification_accuracy(beta_all):
    """Compute accuracy: argmax of predictions vs true labels."""
    beta_stack = np.array(beta_all).T  # (p, K) = (512, 10)
    y_pred_scores = X_test @ beta_stack  # (n_test, K)
    y_pred_classes = np.argmax(y_pred_scores, axis=1)
    acc = np.mean(y_pred_classes == y_test)
    return acc

for ilam, lambda_reg in enumerate(lambda_regs):

    beta_0_all = []
    beta_tilde_all = []
    beta_1_all = []
    beta_1_tune_all = []

    Omega = X_train.T @ X_train / n_train + lambda_reg * np.eye(p)
    M = np.linalg.solve(Omega, X_train.T @ X_train / n_train)

    for i in range(K):
        y_train_i = y_train_multi[:, i]
        beta_0_i = np.linalg.solve(Omega, X_train.T @ y_train_i) / n_train
        beta_0_all.append(beta_0_i)
        beta_tilde_i = M @ beta_0_i
        beta_tilde_all.append(beta_tilde_i)

    # A, B, C, xi_emp calculation using test samples
    A_emp = (1/n_test) * sum(np.linalg.norm(y_test_multi[:, i] - X_test @ beta_0_all[i])**2 for i in range(K))
    B_emp = (1/n_test) * sum(np.linalg.norm(y_test_multi[:, i] - X_test @ beta_tilde_all[i])**2 for i in range(K))
    C_emp = (1/n_test) * sum((y_test_multi[:, i] - X_test @ beta_0_all[i]).T @ (y_test_multi[:, i] - X_test @ beta_tilde_all[i]) for i in range(K))
    xi_emp = (A_emp - C_emp)/(A_emp + B_emp - 2*C_emp)


    # Tuning version
    df_beta_hat = np.trace(M)/n_train
    df_beta_tilde = np.trace(M @ M)/n_train
    A_tune = (1/n_train) * sum(np.linalg.norm(y_train_multi[:, i] - X_train @ beta_0_all[i])**2 for i in range(K)) / (1 - df_beta_hat)**2
    B_tune = (1/n_train) * sum(np.linalg.norm(y_train_multi[:, i] - X_train @ beta_tilde_all[i])**2 for i in range(K)) / (1 - df_beta_tilde)**2
    C_tune = (1/n_train) * sum((y_train_multi[:, i] - X_train @ beta_0_all[i]).T @ (y_train_multi[:, i] - X_train @ beta_tilde_all[i]) for i in range(K)) \
                / ((1 - df_beta_hat)*(1 - df_beta_tilde))
    xi_tune = (A_tune - C_tune)/(A_tune + B_tune - 2*C_tune)

    for i in range(K):
        beta_1_i = (1 - xi_emp) * beta_0_all[i] + xi_emp * beta_tilde_all[i]
        beta_1_all.append(beta_1_i)
        beta_1_tune_i = (1 - xi_tune) * beta_0_all[i] + xi_tune * beta_tilde_all[i]
        beta_1_tune_all.append(beta_1_tune_i)

    # Empirical risks and tuning risk
    R_0 = test_mse(beta_0_all)
    R_tilde = test_mse(beta_tilde_all)
    R_1 = test_mse(beta_1_all)
    R_1_tune = test_mse(beta_1_tune_all)

    # Training risks
    train_R_0 = train_mse(beta_0_all)
    train_R_tilde_0 = train_mse(beta_tilde_all)
    train_R_1 = train_mse(beta_1_all)
    train_R_1_tune = train_mse(beta_1_tune_all)

    # add the xi and risks to array
    xi_emp_arr.append(xi_emp)
    xitune_arr.append(xi_tune)
    Aemp_arr.append(A_emp)
    Bemp_arr.append(B_emp)
    Cemp_arr.append(C_emp)
    R0_arr.append(R_0)
    Rtilde_arr.append(R_tilde)
    R1_arr.append(R_1)
    R1_tune_arr.append(R_1_tune)
    train_R0_arr.append(train_R_0)
    train_Rtilde_arr.append(train_R_tilde_0)
    train_R1_arr.append(train_R_1)
    train_R1_tune_arr.append(train_R_1_tune)

    # Compute and store test accuracies for the 3 betas
    acc0_arr.append(classification_accuracy(beta_0_all))
    acc_tilde_arr.append(classification_accuracy(beta_tilde_all))
    acc1_arr.append(classification_accuracy(beta_1_all))

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

ax.set_title('CIFAR100', fontsize=ftsize + 4)

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