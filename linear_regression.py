"""
BME-336546-C03 - Linear Regression Tutorial
=============================================
Prediction of health insurance costs based on explanatory variables
using linear regression applied on different models.

This tutorial covers:
  1. Data loading & dummy coding for categorical variables
  2. Data preprocessing & exploration (statistics, histograms, scatter plots, correlation)
  3. Closed-form linear regression (pseudoinverse)
  4. Stochastic Gradient Descent (SGD)
  5. Linear regression using scikit-learn
  6. Polynomial feature transformation + linear regression

Authors: Moran Davoodi, Yuval Ben Sason & Kevin Kotzen
"""

# ============================================================================
# Imports
# ============================================================================
from __future__ import division, print_function, unicode_literals

import numpy as np
import pandas as pd
import seaborn as sns
import random
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

random.seed(10)

# Resolve paths relative to this script's directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Create a folder to save all plots
PLOTS_DIR = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Global counter for auto-naming saved figures
_plot_counter = 0

def save_and_show(filename=None):
    """Save the current figure to PLOTS_DIR, then show it."""
    global _plot_counter
    _plot_counter += 1
    if filename is None:
        filename = f"plot_{_plot_counter:02d}.png"
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches='tight')
    print(f"  -> Saved: plots/{filename}")
    plt.close()  # close figure to free memory and avoid blocking

# ============================================================================
# Helper function: plot ground truth vs prediction
# ============================================================================
def plot_gt_vs_pred(gt_array, pred_array, save_name=None):
    """Plot ground truth vs prediction for train and test sets,
    along with error histograms."""
    title = ['Train', 'Train', 'Test', 'Test']
    plot_vars = [(gt_array[0], pred_array[0]), (gt_array[1], pred_array[1])]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for idx, ax in enumerate(axes.flatten()):
        gt, pred = plot_vars[idx >= 2]
        if np.mod(idx, 2) == 0:
            ax.scatter(np.arange(len(gt)), gt, label='ground truth')
            ax.scatter(np.arange(len(gt)), pred, label='prediction')
            ax.legend()
            ax.set_xlabel('# of beneficiary')
            ax.set_ylabel('Charges [$]')
            ax.set_title(title[idx])
        else:
            sns.histplot(gt - pred, ax=ax, kde=True, fill=True, alpha=0.3, linewidth=0)
            ax.set_title(title[idx])
            ax.set_xlabel('ground truth - prediction')
            ax.set_ylabel('pdf')
    plt.tight_layout()
    save_and_show(save_name)


# ============================================================================
# Helper function: plot linear fit in 3D (feature domain)
# ============================================================================
def plot_lin(pol_data, x_y_axis, y_test, lin_reg):
    """3D visualization of original data vs linear fit in feature domain."""
    x1 = pol_data['x1']
    x2 = pol_data['x2']
    z = pol_data['z']
    data = pol_data['data']
    fig = plt.figure(figsize=(16, 10))

    # --- left subplot: original surface ---
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(x1, x2, z, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False, alpha=0.5)
    R = np.random.randint(0, z.shape[0] * z.shape[1], (100,))
    ax.scatter(x1.flatten()[R], x2.flatten()[R], data.flatten()[R])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    ax.set_zlabel('$y$')
    ax.view_init(elev=10, azim=140)
    ax.set_title('Original')

    # --- right subplot: linear fit ---
    x1m, x2m = np.meshgrid(np.sort(x_y_axis[:, 0]), np.sort(x_y_axis[:, 1]))
    z_fit = lin_reg.intercept_ + lin_reg.coef_[0] * x1m + lin_reg.coef_[1] * x2m
    R = np.random.randint(0, y_test.shape[0], (100,))
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(x1m, x2m, z_fit, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False, alpha=0.5)
    ax.scatter(x_y_axis[:, 0].flatten()[R], x_y_axis[:, 1].flatten()[R],
               y_test.flatten()[R])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    ax.set_zlabel('$y$')
    ax.view_init(elev=10, azim=140)
    ax.set_title('Linear fit in feature domain')
    ax.set_zlim(0, 200)
    plt.tight_layout()
    save_and_show("linear_fit_3d.png")


# ============================================================================
# Helper function: plot polynomial fit in 3D (transformed feature domain)
# ============================================================================
def plot_pol(pol_data, rel_x_y_axis, y_test, lin_reg, rel_indices, x_label, y_label):
    """3D visualization of original data vs polynomial fit in transformed domain."""
    x1 = pol_data['x1']
    x2 = pol_data['x2']
    z = pol_data['z']
    data = pol_data['data']
    fig = plt.figure(figsize=(16, 10))

    # --- left subplot: original surface ---
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(x1, x2, z, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False, alpha=0.5)
    R = np.random.randint(0, z.shape[0] * z.shape[1], (100,))
    ax.scatter(x1.flatten()[R], x2.flatten()[R], data.flatten()[R])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    ax.set_zlabel('$y$')
    ax.view_init(elev=10, azim=140)
    ax.set_title('Original')

    # --- right subplot: polynomial fit in transformed domain ---
    x1m, x2m = np.meshgrid(np.sort(rel_x_y_axis[:, 0]),
                            np.sort(rel_x_y_axis[:, 1]))
    z_fit = (lin_reg.coef_[0]
             + x1m * lin_reg.coef_[rel_indices[0]]
             + x2m * lin_reg.coef_[rel_indices[1]])
    R = np.random.randint(0, y_test.shape[0], (100,))
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(x1m, x2m, z_fit, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False, alpha=0.5)
    ax.scatter(rel_x_y_axis[:, 0].flatten()[R],
               rel_x_y_axis[:, 1].flatten()[R],
               y_test.flatten()[R])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.set_zlabel('$y$')
    ax.view_init(elev=10, azim=125)
    ax.set_title('Linear fit in transformed feature domain')
    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(0, 100)
    ax.set_zlim3d(0, 200)
    plt.tight_layout()
    save_and_show("polynomial_fit_3d.png")


# ============================================================================
# PART 1: Data Loading
# ============================================================================
print("=" * 70)
print("PART 1: DATA LOADING")
print("=" * 70)

X = pd.read_csv(SCRIPT_DIR / "data" / "insurance.csv")
print("\n--- Raw data sample ---")
print(X.sample(8, random_state=5))

# ============================================================================
# PART 2: Dummy Coding (one-hot encoding for categorical variables)
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: DUMMY CODING")
print("=" * 70)

X = pd.get_dummies(data=X, drop_first=True, dtype=float)
print("\n--- Data after dummy coding ---")
print(X.sample(8, random_state=5))

# ============================================================================
# PART 3: Data Preprocessing & Exploration
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: DATA EXPLORATION")
print("=" * 70)

# 3.1 Summary statistics
print("\n--- Summary Statistics ---")
print(X.describe())

# 3.2 Histograms of every variable
X.hist(figsize=(12, 10), bins=30)
plt.suptitle("Distribution of all variables", fontsize=16)
plt.tight_layout()
save_and_show("01_histograms.png")

# 3.3 Scatter plot: Age vs Charges
plt.figure(figsize=(8, 6))
plt.scatter(X['age'], X['charges'], alpha=0.5)
plt.xlabel('Age [years]')
plt.ylabel('Charges [$]')
plt.title('Age vs Charges')
save_and_show("02_age_vs_charges.png")

# 3.4 Age vs Charges colored by smoking status (shows bimodal groups)
plt.figure(figsize=(8, 6))
colors = X['smoker_yes'].map({1: 'red', 0: 'blue'})
plt.scatter(X['age'], X['charges'], c=colors, alpha=0.5)
plt.xlabel('Age [years]')
plt.ylabel('Charges [$]')
plt.title('Age vs Charges (colored by smoker status)')
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',  label='Smoker'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Non-smoker'),
])
save_and_show("03_age_vs_charges_smoker.png")

# 3.5 BMI vs Charges colored by smoking status
plt.figure(figsize=(8, 6))
plt.scatter(X['bmi'], X['charges'], c=colors, alpha=0.5)
plt.xlabel('BMI [kg/m²]')
plt.ylabel('Charges [$]')
plt.title('BMI vs Charges (colored by smoker status)')
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',  label='Smoker'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Non-smoker'),
])
save_and_show("04_bmi_vs_charges_smoker.png")

# 3.6 Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.tight_layout()
save_and_show("05_correlation_heatmap.png")

# ============================================================================
# PART 4: Closed-Form Linear Regression (Pseudoinverse)
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: CLOSED-FORM LINEAR REGRESSION (Pseudoinverse)")
print("=" * 70)

# Prepare X and y
y = X['charges']
X.drop(columns='charges', inplace=True)
X = X.to_numpy()
y = y.to_numpy()
X = np.concatenate((np.ones((len(y), 1)), X), axis=1)  # add bias term

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# --- Closed-form solution: w = (X^T X)^{-1} X^T y ---
w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Predictions
y_pred_train = X_train @ w
y_pred_test  = X_test  @ w

print(f"Weights (w): {w}")

gt_array   = [y_train, y_test]
pred_array = [y_pred_train, y_pred_test]
plot_gt_vs_pred(gt_array, pred_array, "06_closed_form_gt_vs_pred.png")

# ============================================================================
# PART 5: Stochastic Gradient Descent (SGD)
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: SGD LINEAR REGRESSION")
print("=" * 70)

# 5.1 Standardization (fit on train, transform both train & test)
scaler = StandardScaler()
X_train[:, 1:] = scaler.fit_transform(X_train[:, 1:])   # skip bias column
X_test[:, 1:]  = scaler.transform(X_test[:, 1:])
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test  = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

# 5.2 SGD implementation
alpha = 0.005

w1 = np.random.randn(X_train.shape[1])  # random initialization
mse_train = []
mse_test  = []

for x_i, y_i in zip(X_train, y_train):
    # Gradient for single sample: -(y_i - w^T x_i) * x_i
    error = y_i - w1 @ x_i
    gradient = -error * x_i
    w1 = w1 - alpha * gradient

    # Record MSE on full train and test sets
    mse_train.append(np.mean((y_train - X_train @ w1) ** 2))
    mse_test.append(np.mean((y_test  - X_test  @ w1) ** 2))

# Plot learning curve
plt.figure(figsize=(8, 5))
plt.plot(np.arange(len(mse_train)), mse_train)
plt.plot(np.arange(len(mse_test)),  mse_test)
plt.legend(("train", "test"))
plt.xlabel("iteration #")
plt.ylabel("MSE")
plt.title("SGD Learning Curve")
save_and_show("07_sgd_learning_curve.png")

# SGD predictions
y_pred_train = X_train @ w1
y_pred_test  = X_test  @ w1

gt_array   = [y_train, y_test]
pred_array = [y_pred_train, y_pred_test]
plot_gt_vs_pred(gt_array, pred_array, "08_sgd_gt_vs_pred.png")

# ============================================================================
# PART 6: Linear Regression with scikit-learn (on scaled data)
# ============================================================================
print("\n" + "=" * 70)
print("PART 6: SKLEARN LINEAR REGRESSION")
print("=" * 70)

# fit_intercept=False because X already has a bias column of ones
lin_reg_sklearn = LinearRegression(fit_intercept=False)
lin_reg_sklearn.fit(X_train, y_train)

y_pred_train = lin_reg_sklearn.predict(X_train)
y_pred_test  = lin_reg_sklearn.predict(X_test)

gt_array   = [y_train, y_test]
pred_array = [y_pred_train, y_pred_test]
plot_gt_vs_pred(gt_array, pred_array, "09_sklearn_gt_vs_pred.png")

# ============================================================================
# PART 7: Polynomial Features + Linear Regression
# ============================================================================
print("\n" + "=" * 70)
print("PART 7: POLYNOMIAL FEATURES + LINEAR REGRESSION")
print("=" * 70)

# 7.1 Load new (2-feature) dataset
pol_data = np.load(SCRIPT_DIR / "data" / "pol_data.npz")
X_new = np.c_[pol_data['x1'].ravel(), pol_data['x2'].ravel()]
y = pol_data['data'].flatten()

X_train, X_test, y_train, y_test = train_test_split(
    X_new, y, test_size=0.2, random_state=10
)

# 7.2 Linear fit on original (unscaled) features
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_train_lin = lin_reg.predict(X_train)
y_pred_test_lin  = lin_reg.predict(X_test)

gt_array   = [y_train, y_test]
pred_array = [y_pred_train_lin, y_pred_test_lin]
print("\n--- Linear fit in original domain ---")
plot_gt_vs_pred(gt_array, pred_array, "10_linear_fit_gt_vs_pred.png")

# 7.3 3D visualization of linear fit
plot_lin(pol_data, X_test, y_test, lin_reg)  # saves as 11_linear_fit_3d.png

# 7.4 Polynomial feature transformation (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly  = poly.transform(X_test)

# 7.5 Linear regression on transformed features
lin_reg = LinearRegression()
lin_reg.fit(X_train_poly, y_train)

y_pred_train_pol = lin_reg.predict(X_train_poly)
y_pred_test_pol  = lin_reg.predict(X_test_poly)

gt_array   = [y_train, y_test]
pred_array = [y_pred_train_pol, y_pred_test_pol]
print("\n--- Polynomial fit (degree 2) ---")
plot_gt_vs_pred(gt_array, pred_array, "12_poly_fit_gt_vs_pred.png")

# 7.6 Identify the two most significant polynomial coefficients
print("\nPolynomial feature names:", poly.get_feature_names_out())
print("Coefficients:", lin_reg.coef_)
print("Intercept:",    lin_reg.intercept_)

# The two most significant coefficients correspond to x1*x2 and x2^2
# PolynomialFeatures(degree=2) produces: [x1, x2, x1^2, x1*x2, x2^2]
# Indices (0-based in coef_): x1*x2 -> index 3, x2^2 -> index 4
rel_indices = [3, 4]   # indices of x1*x2 and x2^2 in coef_
x_label = r'$x_1 x_2$'
y_label = r'$x_2^2$'

# Extract the two relevant transformed features for test set
rel_x = X_test_poly[:, rel_indices]

# 7.7 3D visualization of polynomial fit
plot_pol(pol_data, rel_x, y_test, lin_reg, rel_indices, x_label, y_label)  # saves as 13_polynomial_fit_3d.png

# ============================================================================
# CONCLUSIONS
# ============================================================================
print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)
print("""
In this tutorial we implemented linear regression in several ways:
  - Pseudoinverse (closed-form)
  - Stochastic Gradient Descent (SGD)
  - scikit-learn LinearRegression

Key observations:
  - SGD is sensitive to learning rate and requires standardization.
  - Smoking status is the strongest predictor of insurance charges.
  - Linear regression can fit polynomial models via feature transformation.
""")
