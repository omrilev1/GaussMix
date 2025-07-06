import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from tqdm import tqdm 
from utils_linear_mixing import * 
    
# Pick dataset our of the next options:
#   'synthetic_Gaussian', 
#   'synthetic_MLP', 
#   'synthetic_Uniform',
#   'housing',
#   'crime'
#   'tecator'
#   'wine'
#   'bike'
 
dataset_type = 'synthetic_Gaussian' 

C_max, lambda_opt, X_train, y_train,\
    X_test, y_test, \
        lambda_min, lambda_min_XY,\
            n, d = GetDataset(dataset_type)
  
# Epsilon and delta values 
epsilon_values = np.logspace(-1.0, 1.0, 5)
delta_DP = np.min((1e-6, 0.01/n)) 

# linear mixing constant 
tau = np.sqrt(2.0 * np.log(3.0/delta_DP))
       
# baseline: simple OLS (with regularization given by a fixed pre-determined value)    
ridge_model = Ridge(alpha=lambda_opt, fit_intercept=False)  # alpha is the regularization strength
ridge_model.fit(X_train, y_train)
ridge_y_pred = ridge_model.predict(X_test)
baseline_test_mse = np.mean((y_test - ridge_y_pred)**2)
ridge_y_pred_train = ridge_model.predict(X_train)
baseline_train_mse = np.mean((y_train - ridge_y_pred_train)**2)
    
print('============================')
print('baseline test mse is ' + str(baseline_test_mse))
print('baseline train mse is ' + str(baseline_train_mse))
print('============================')

# Hyperparameter: k 
percentage_k = 2.5 # [1.5]
k_val = int(percentage_k * d)  # Three different projection dimensions    

# Number of Monte-Carlo iterations
iters = 250 
print_noise_values = False # change to True if you want to print the noise values
 
# Prepare a list to store MSE for each sigma
mse_for_sigmas                                  = []
mse_for_sigmas_std                              = []
mse_for_sigmas_noise_sheffet                    = []
mse_for_sigmas_std_noise_sheffet                = []
mse_for_sigmas_adassp                           = []
mse_for_sigmas_std_adassp                       = []
mse_for_sigmas_noise_sheffet_our_analysis       = []
mse_for_sigmas_std_noise_sheffet_our_analysis   = []
    
for eps_idx, eps in tqdm(enumerate(epsilon_values)):
    print('Started eps = ' + str(eps))
    curr_test_mse                               = []
    curr_test_mse_noise_sheffet                 = []
    curr_test_mse_adassp                        = [] 
    curr_test_mse_noise_sheffet_our_analysis    = [] 
    # Compute noise values for all methods: 
    ############ Ours ##############
    sigma_DP = 2.0 * np.log(1.25/delta_DP) / (eps**2)
    ratio_sigma1_sigma2 = 1.0
    sigma_matrix = solve_sigma_renyi_full(sigma_DP, k_val, d, n, delta_DP, eps, C_max**2)
    sigma_eigenval = sigma_matrix/np.sqrt(k_val)
    
    ########### Sheffet's method ##############
    sigma_matrix_sheffet = 4.0 * (np.sqrt(2.0 * k_val * np.log(8.0/(delta_DP))) + np.log(8.0/(delta_DP)))/(eps)
    
    ########### Sheffet's method, our bound ##############
    sigma_matrix_sheffet_ours = solve_sigma_renyi(sigma_DP, k_val, d, n, delta_DP, eps/2.0, C_max**2)
        
    ########### AdaSSP: calculate noisy minimum eigenvalue ##############
    _tilde = np.max((0,  + np.sqrt(np.log(6.0/delta_DP))/(eps/3.0) * np.random.randn() / 
                              - (np.log(6.0/delta_DP))/(eps/3.0)))
    lambda_adassp = np.max((0, np.sqrt(d * np.log(6.0/delta_DP) * np.log(2.0*(d**2)/(0.0001)))/(eps/3.0) - _tilde))
        
    if print_noise_values:
        print('sigma_DP is: ' + str(n * d * sigma_DP))
        print('sigma_matrix is: ' + str(k_val * d * sigma_matrix))
        print('sigma_matrix sheffet is: ' + str(k_val * d * sigma_matrix_sheffet))
        print('Their ratio (DP/Matrix) is: ' + str((n/k_val) * sigma_DP/sigma_matrix))
            
    for iter in tqdm(range(iters)):
        
        # generate noises and random matrices 
        y_train_vec = y_train.reshape(-1, 1)  # (n,1)
        
        S = np.random.randn(k_val, n) 
        N_ours = np.random.randn(k_val, d)
        N_ours_y = np.random.randn(k_val)
        
        S_sheffet = np.random.randn(k_val, n)
        N_Sheffet = np.random.randn(k_val, d)
        N_Sheffet_y = np.random.randn(k_val)
        
        ############################### our method ###############################
        if sigma_matrix <= tau:
            X_train_full_PR = S @ X_train + np.sqrt(sigma_matrix) * N_ours  
            y_PR = (S @ y_train_vec).ravel() + np.sqrt(sigma_matrix) * N_ours_y
        else:        
            gamma_tilde = np.max((0, lambda_min_XY - np.sqrt(sigma_eigenval) * (tau - np.random.randn())))
            sigma_tilde = np.sqrt(np.max((0, sigma_matrix - gamma_tilde)))
            X_train_full_PR = S @ X_train + sigma_tilde * N_ours  # (n_prime,d)
            y_PR = (S @ y_train_vec).ravel() + sigma_tilde * N_ours_y
            
        theta_ridge_private = np.linalg.inv(X_train_full_PR.T @ X_train_full_PR) @ X_train_full_PR.T @ y_PR
            
        ###############################  Sheffet's method ###############################
        # Compensate factor 2 from zero out vs. add/remove 
        W2 = (4.0/(eps))* (np.sqrt(2.0 * k_val * np.log(8.0/(delta_DP))) + np.log(8.0/(delta_DP)))

        z = np.random.laplace(loc=0.0, scale=4.0/(eps))
        
        if lambda_min_XY > W2 + z + 4.0 * np.log(1/(delta_DP))/(eps):
            X_train_full_PR_sheffet = S_sheffet @ X_train
            y_PR_sheffet = (S_sheffet @ y_train_vec).ravel()
        else:
            X_train_full_PR_sheffet = S_sheffet @ X_train + np.sqrt(sigma_matrix_sheffet) * N_Sheffet  
            y_PR_sheffet = (S_sheffet @ y_train_vec).ravel() + np.sqrt(sigma_matrix_sheffet/ratio_sigma1_sigma2) * N_Sheffet_y
        
        theta_ridge_private_sheffet = np.linalg.inv(X_train_full_PR_sheffet.T @ X_train_full_PR_sheffet) @ X_train_full_PR_sheffet.T @ y_PR_sheffet
            
        ###############################  Sheffet's method, our's noise ###############################
        # Compensate factor 2 from zero out vs. add/remove 
        W2 = sigma_matrix_sheffet_ours/2
        z = np.random.laplace(loc=0.0, scale=4.0/(eps))
        
        if lambda_min_XY > W2 + z + 4.0 * np.log(1/(delta_DP))/(eps):
            X_train_full_PR_sheffet = S_sheffet @ X_train
            y_PR_sheffet = (S_sheffet @ y_train_vec).ravel()
        else:
            X_train_full_PR_sheffet = S_sheffet @ X_train + np.sqrt(sigma_matrix_sheffet_ours) * N_Sheffet  
            y_PR_sheffet = (S_sheffet @ y_train_vec).ravel() + np.sqrt(sigma_matrix_sheffet_ours) * N_Sheffet_y
            
        theta_ridge_private_sheffet_ours = np.linalg.inv(X_train_full_PR_sheffet.T @ X_train_full_PR_sheffet) @ X_train_full_PR_sheffet.T @ y_PR_sheffet
            
        ############################### adassp ###############################
        N_upper = np.random.randn(d, d)
        N = np.triu(N_upper)  # Upper triangular part
        N = N + N.T - np.diag(np.diag(N))  # Make symmetric
        XTX_noisy = X_train.T @ X_train + (np.sqrt(np.log(6.0/delta_DP))/(eps/3.0)) * N  
        XTy_noisy = X_train.T @ y_train + C_max * (np.sqrt(np.log(6.0/delta_DP))/(eps/3.0)) * np.random.randn(d)  
        theta_adassp = np.linalg.inv(XTX_noisy + lambda_adassp*np.eye(d)) @ XTy_noisy
            
        ############################### Evaluate ###############################
        # Evaluate on original test data
        ridge_y_pred = X_test @ theta_ridge_private
        curr_test_mse.append(np.mean((y_test - ridge_y_pred)**2))

        ridge_y_pred_sheffet = X_test @ theta_ridge_private_sheffet
        curr_test_mse_noise_sheffet.append(np.mean((y_test - ridge_y_pred_sheffet)**2))            

        ridge_y_pred_sheffet = X_test @ theta_ridge_private_sheffet_ours
        curr_test_mse_noise_sheffet_our_analysis.append(np.mean((y_test - ridge_y_pred_sheffet)**2))           
        
        ridge_y_pred_adassp = X_test @ theta_adassp
        curr_test_mse_adassp.append(np.real(np.mean((y_test - ridge_y_pred_adassp)**2)))
            

    # MSEs and confidence intervals 
    mse_for_sigmas.append(np.mean(curr_test_mse))
    mse_for_sigmas_std.append(1.96 * np.std(curr_test_mse)/np.sqrt(iters))
    mse_for_sigmas_noise_sheffet.append(np.mean(curr_test_mse_noise_sheffet))
    mse_for_sigmas_std_noise_sheffet.append(1.96 * np.std(curr_test_mse_noise_sheffet)/np.sqrt(iters))
    mse_for_sigmas_noise_sheffet_our_analysis.append(np.mean(curr_test_mse_noise_sheffet_our_analysis))
    mse_for_sigmas_std_noise_sheffet_our_analysis.append(1.96 * np.std(curr_test_mse_noise_sheffet_our_analysis)/np.sqrt(iters))
    mse_for_sigmas_adassp.append(np.mean(curr_test_mse_adassp))
    mse_for_sigmas_std_adassp.append(1.96 * np.std(curr_test_mse_adassp)/np.sqrt(iters))

    print('====================')  
    print('Test MSE: ' + str(np.mean(curr_test_mse)))
    print('Test MSE Sheffet: ' + str(np.mean(curr_test_mse_noise_sheffet)))
    print('Test MSE Sheffet, our bound: ' + str(np.mean(curr_test_mse_noise_sheffet_our_analysis)))
    print('Test MSE ADASSP: ' + str(np.mean(curr_test_mse_adassp)))
        
################################# Generate Plots #################################
plt.figure(figsize=(10, 6))

# Plot baseline (non-private)
plt.axhline(y=baseline_test_mse, color='black', linestyle='--', label='Baseline')

# Extract the single n_prime value
label_suffix = fr"$\frac{{k}}{{d}} = $ {percentage_k:.3f}"

# Sheffet '17 (original noise)
plt.plot(epsilon_values, mse_for_sigmas_noise_sheffet, color='black', marker='p',
         label=fr"[Alg. 1, Sheffet '17]: {label_suffix}")
plt.fill_between(epsilon_values,
                 np.array(mse_for_sigmas_noise_sheffet) - np.array(mse_for_sigmas_std_noise_sheffet),
                 np.array(mse_for_sigmas_noise_sheffet) + np.array(mse_for_sigmas_std_noise_sheffet),
                 color='black', alpha=0.2)

# ADASSP baseline
plt.plot(epsilon_values, mse_for_sigmas_adassp, color='blue', marker='*', markersize=12,
         label="ADASSP [Wang '18]")
plt.fill_between(epsilon_values,
                 np.array(mse_for_sigmas_adassp) - np.array(mse_for_sigmas_std_adassp),
                 np.array(mse_for_sigmas_adassp) + np.array(mse_for_sigmas_std_adassp),
                 color='blue', alpha=0.2)

# Sheffet '17 with our analysis
plt.plot(epsilon_values, mse_for_sigmas_noise_sheffet_our_analysis, color='magenta', marker='p',
         label=fr"[Alg. 1, Sheffet '17] (ours): {label_suffix}")
plt.fill_between(epsilon_values,
                 np.array(mse_for_sigmas_noise_sheffet_our_analysis) - np.array(mse_for_sigmas_std_noise_sheffet_our_analysis),
                 np.array(mse_for_sigmas_noise_sheffet_our_analysis) + np.array(mse_for_sigmas_std_noise_sheffet_our_analysis),
                 color='magenta', alpha=0.2)

# Our method: Linear mixing
plt.plot(epsilon_values, mse_for_sigmas, color='orangered', marker='o', markersize=12,
         label=fr"Linear mixing (ours): {label_suffix}")
plt.fill_between(epsilon_values,
                 np.array(mse_for_sigmas) - np.array(mse_for_sigmas_std),
                 np.array(mse_for_sigmas) + np.array(mse_for_sigmas_std),
                 color='orangered', alpha=0.2)

# Axis labels and scale
plt.xlabel(r"$\epsilon_{\mathrm{DP}}$", fontsize=20)
plt.ylabel("Test MSE", fontsize=20)
plt.xscale('log')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)

# Dataset-dependent title
dataset_titles = {
    'crime': "Communities & Crime dataset",
    'housing': "Boston Housing dataset",
    'bike': "Bike Sharing dataset",
    'wine': "Wine dataset",
    'tecator': "Tecator dataset",
    'synthetic_MLP': "Synthetic dataset",
    'synthetic_Gaussian': "Gaussian dataset",
    'synthetic_Uniform': "Uniform dataset"
}
plt.title(dataset_titles.get(dataset_type, ""), fontsize=24)

# DP delta annotation
textstr = r"$\delta_{\mathrm{DP}} = \min(10^{-6}, \frac{0.01}{n})$"
plt.text(0.98, 0.98, textstr, transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

# Legend
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.3),
           ncol=2, fontsize=18, frameon=False)

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/linear_mixing_{dataset_type}.pdf", bbox_inches='tight')
