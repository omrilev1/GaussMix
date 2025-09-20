import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from tqdm import tqdm 
from utils_linear_mixing import * 
import random
import warnings
warnings.filterwarnings("ignore")  # suppress all warnings from all modules

random.seed(50)
np.random.seed(50)

# Full set of datasets to run our algorithms 
# datasets_run_list = ['housing', 'crime', 'tecator', 'wine', 'bike', 'autompg', 'energy', 'concrete', 'elevators', 'Gas', \
#                      'airfoil', 'breastcancer', 'parkinsons', 'sml', 'keggundirected', 'protein', 'tamielectric', \
#                      'keggdirected', 'yacht', 'solar', '3droad', 'slice', 'servo', 'autos', 'concreteslump', 'fertility', 'forest',\
#                      'houseelectric', 'kin40k', 'machine', 'pol', 'pendulum', 'pumadyn32nm', 'buzz', \
#                      'synthetic_MLP', 'synthetic_Gaussian', 'synthetic_Uniform', 'synthetic_eig']

# Pick target datasets to run from the previous list and put them inside dataset_run_list 
datasets_run_list = ['tecator']

# Set to True if want to add a vertical line at the predicted eps_th
add_vertical_line = False 

# Hyperparameters
target_varrho = 1e-10
percentage_k = 2.5    # overall k is going to be percentage_k * max(d, log(1/varrho))

# Number of Monte-Carlo iterations
iters = 250 

# To print intermediate nise values and data properties 
print_noise_values = False              # change to True if you want to print the noise values
print_singular_values_values = False    # change to True if you want to print the noise values

# epsilon values to run our algorithm 
epsilon_values = np.logspace(-1.2, 1.5, 6)

for dataset_type in datasets_run_list:
    print('Running dataset ' + dataset_type)

    C_max, X_train, y_train,\
        X_test, y_test, \
            lambda_min, lambda_min_XY,\
                n, d = GetDataset(dataset_type)

    print('=============================')
    print('lambda min is ' + str(lambda_min)
          + ', lambda min XY is ' + str(lambda_min_XY))
    print('n is ' + str(n) + ', d is ' + str(d))
    print('=============================')

    # Epsilon and delta values 
    delta_DP = 1/(n**2)

    # linear mixing constant 
    tau = np.sqrt(2.0 * np.log(3.0/delta_DP))

    # baseline: simple Ridge regression
    lambda_baseline = 1e-16 
    ridge_model = Ridge(alpha=lambda_baseline, fit_intercept=False)  # alpha is the regularization strength
    ridge_model.fit(X_train, y_train)
    ridge_y_pred = ridge_model.predict(X_test)
    baseline_test_mse = np.sum((y_test - ridge_y_pred)**2)
    ridge_y_pred_train = ridge_model.predict(X_train)
    baseline_train_mse = np.sum((y_train - ridge_y_pred_train)**2)
    norm_theta_star = (ridge_model.coef_).T @ (ridge_model.coef_)
    theta_star = ridge_model.coef_

    # Simple OLS error using sklearn's LinearRegression (no regularization at all)
    ols_model = LinearRegression(fit_intercept=False)
    ols_model.fit(X_train, y_train)
    ols_y_pred = ols_model.predict(X_train)
    ols_train_mse = np.sum((y_train - ols_y_pred)**2)

    print('============================')
    print('baseline test mse (Ridge ~ OLS) is ' + str(baseline_test_mse/n))
    print('baseline train mse (Ridge ~ OLS) is ' + str(baseline_train_mse/n))
    print('simple OLS train mse is ' + str(ols_train_mse/n))
    print('============================')

    # Set k_value 
    k_val = np.max((int(percentage_k * d), int(percentage_k * np.log(1/target_varrho))))  # target_rho is the failure probability of the random projection

    # Prepare a list to store MSE for each sigma
    mse_for_sigmas                                      = []
    mse_for_sigmas_std                                  = []
    mse_for_sigmas_noise_sheffet                        = []
    mse_for_sigmas_std_noise_sheffet                    = []
    mse_for_sigmas_adassp                               = []
    mse_for_sigmas_std_adassp                           = []
    mse_for_sigmas_noise_sheffet_our_analysis           = []
    mse_for_sigmas_std_noise_sheffet_our_analysis       = []

    train_mse_for_sigmas                                = []
    train_mse_for_sigmas_std                            = []
    train_mse_for_sigmas_noise_sheffet                  = []
    train_mse_for_sigmas_std_noise_sheffet              = []
    train_mse_for_sigmas_adassp                         = []
    train_mse_for_sigmas_std_adassp                     = []
    train_mse_for_sigmas_noise_sheffet_our_analysis     = []
    train_mse_for_sigmas_std_noise_sheffet_our_analysis = []

    for eps_idx, eps in tqdm(enumerate(epsilon_values)):
        print('Started eps = ' + str(eps))
        curr_test_mse                               = []
        curr_test_mse_noise_sheffet                 = []
        curr_test_mse_adassp                        = [] 
        curr_test_mse_noise_sheffet_our_analysis    = [] 
        curr_train_mse                              = []
        curr_train_mse_noise_sheffet                = []
        curr_train_mse_adassp                       = [] 
        curr_train_mse_noise_sheffet_our_analysis   = [] 

        # Compute noise values for all methods: 
        ############ Ours ##############
        sigma_DP = 2.0 * np.log(1.25/delta_DP) / (eps**2)
        ratio_sigma1_sigma2 = 1.0
        sigma_matrix = solve_sigma_renyi_full(sigma_DP, k_val, d, n, delta_DP, eps, C_max**2)
        print('simple OLS train mse normalized by sigma^2 + lambda_min is ' + str(ols_train_mse/(sigma_matrix + lambda_min_XY)))
        if eps_idx == 2:
            # store value to add in the figure title 
            R_over_gamma_plot = ols_train_mse/(sigma_matrix + lambda_min_XY)
            
        # Ridge regression with regularization of strength sigma_matrix 
        ridge_model = Ridge(alpha=sigma_matrix, fit_intercept=False)  # alpha is the regularization strength
        ridge_model.fit(X_train, y_train)
        ridge_y_pred_train = ridge_model.predict(X_train)
        train_mse_ridge_sigma = np.mean((y_train - ridge_y_pred_train)**2)
        norm_coeff = (ridge_model.coef_.T @ ridge_model.coef_) * sigma_matrix

        sigma_eigenval = sigma_matrix/np.sqrt(k_val)
        print('Current sigma is: ' + str(sigma_matrix))

        ########### Sheffet's method ##############
        sigma_matrix_sheffet = 4.0 * (np.sqrt(2.0 * k_val * np.log(8.0/(delta_DP))) + np.log(8.0/(delta_DP)))/(eps)

        ########### Sheffet's method, our bound ##############
        sigma_matrix_sheffet_ours = solve_sigma_renyi(sigma_DP, k_val, d, n, delta_DP, eps/2.0, C_max**2)

        ########### AdaSSP: calculate noisy minimum eigenvalue ##############
        lambda_min_tilde = np.max((0, lambda_min + np.sqrt(np.log(6.0/delta_DP))/(eps/3.0) * np.random.randn() / 
                                  - (np.log(6.0/delta_DP))/(eps/3.0)))
        lambda_adassp = np.max((0, np.sqrt(d * np.log(6.0/delta_DP) * np.log(2.0*(d**2)/(target_varrho)))/(eps/3.0) - lambda_min_tilde))

        bound_adassp = (C_max + norm_theta_star) * np.log(6.0/delta_DP) * np.log(2.0*(d**2)/(target_varrho))/(eps)**2 \
            * np.trace(np.linalg.inv(X_train.T @ X_train + lambda_adassp * np.eye(d))) + \
                (lambda_adassp**2) * theta_star.T @ np.linalg.inv(X_train.T @ X_train + lambda_adassp * np.eye(d)) @ theta_star

        print('Baseline AdaSSP Upper Bound is ' + str(bound_adassp))
        print('Baseline MSE for Ridge with sigma is ' + str(train_mse_ridge_sigma))

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
            ridge_y_pred = X_train @ theta_ridge_private
            curr_train_mse.append(np.mean((y_train - ridge_y_pred)**2))

            ridge_y_pred_sheffet = X_test @ theta_ridge_private_sheffet
            curr_test_mse_noise_sheffet.append(np.mean((y_test - ridge_y_pred_sheffet)**2))  
            ridge_y_pred_sheffet = X_train @ theta_ridge_private_sheffet          
            curr_train_mse_noise_sheffet.append(np.mean((y_train - ridge_y_pred_sheffet)**2))

            ridge_y_pred_sheffet = X_test @ theta_ridge_private_sheffet_ours
            curr_test_mse_noise_sheffet_our_analysis.append(np.mean((y_test - ridge_y_pred_sheffet)**2))           
            ridge_y_pred_sheffet = X_train @ theta_ridge_private_sheffet_ours
            curr_train_mse_noise_sheffet_our_analysis.append(np.mean((y_train - ridge_y_pred_sheffet)**2))           

            ridge_y_pred_adassp = X_test @ theta_adassp
            curr_test_mse_adassp.append(np.mean((y_test - ridge_y_pred_adassp)**2))
            ridge_y_pred_adassp = X_train @ theta_adassp
            curr_train_mse_adassp.append(np.mean((y_train - ridge_y_pred_adassp)**2))

        # MSEs and confidence intervals 
        mse_for_sigmas.append(np.mean(curr_test_mse))
        mse_for_sigmas_std.append(1.96 * np.std(curr_test_mse)/np.sqrt(iters))
        mse_for_sigmas_noise_sheffet.append(np.mean(curr_test_mse_noise_sheffet))
        mse_for_sigmas_std_noise_sheffet.append(1.96 * np.std(curr_test_mse_noise_sheffet)/np.sqrt(iters))
        mse_for_sigmas_noise_sheffet_our_analysis.append(np.mean(curr_test_mse_noise_sheffet_our_analysis))
        mse_for_sigmas_std_noise_sheffet_our_analysis.append(1.96 * np.std(curr_test_mse_noise_sheffet_our_analysis)/np.sqrt(iters))
        mse_for_sigmas_adassp.append(np.mean(curr_test_mse_adassp))
        mse_for_sigmas_std_adassp.append(1.96 * np.std(curr_test_mse_adassp)/np.sqrt(iters))

        train_mse_for_sigmas.append(np.mean(curr_train_mse))
        train_mse_for_sigmas_std.append(1.96 * np.std(curr_train_mse)/np.sqrt(iters))
        train_mse_for_sigmas_noise_sheffet.append(np.mean(curr_train_mse_noise_sheffet))
        train_mse_for_sigmas_std_noise_sheffet.append(1.96 * np.std(curr_train_mse_noise_sheffet)/np.sqrt(iters))
        train_mse_for_sigmas_noise_sheffet_our_analysis.append(np.mean(curr_train_mse_noise_sheffet_our_analysis))
        train_mse_for_sigmas_std_noise_sheffet_our_analysis.append(1.96 * np.std(curr_train_mse_noise_sheffet_our_analysis)/np.sqrt(iters))
        train_mse_for_sigmas_adassp.append(np.mean(curr_train_mse_adassp))
        train_mse_for_sigmas_std_adassp.append(1.96 * np.std(curr_train_mse_adassp)/np.sqrt(iters))

        print('====================')  
        print('Test MSE: ' + str(np.mean(curr_test_mse)))
        print('Test MSE Sheffet: ' + str(np.mean(curr_test_mse_noise_sheffet)))
        print('Test MSE Sheffet, our bound: ' + str(np.mean(curr_test_mse_noise_sheffet_our_analysis)))
        print('Test MSE ADASSP: ' + str(np.mean(curr_test_mse_adassp)))

        print('====================')  
        print('Train MSE: ' + str(np.mean(curr_train_mse)))
        print('Train MSE Sheffet: ' + str(np.mean(curr_train_mse_noise_sheffet)))
        print('Train MSE Sheffet, our bound: ' + str(np.mean(curr_train_mse_noise_sheffet_our_analysis)))
        print('Train MSE ADASSP: ' + str(np.mean(curr_train_mse_adassp)))

    ################################# Generate Plots #################################
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    # Extract the single n_prime value
    label_suffix = r"$\frac{k}{\max\{d,\log(1/\varrho)\}} = " + f"{percentage_k:.3f}$"

    # Define colors and markers for consistency
    colors = {
        'sheffet': 'black',
        'adassp': 'blue', 
        'sheffet_ours': 'magenta',
        'linear_mixing': 'orangered'
    }

    markers = {
        'sheffet': 'p',
        'adassp': '*',
        'sheffet_ours': 'p', 
        'linear_mixing': 'o'
    }

    # Train MSE variables (assuming they follow the pattern of replacing 'test' with 'train' in variable names)
    train_mse_for_sigmas_noise_sheffet = locals().get('train_mse_for_sigmas_noise_sheffet', mse_for_sigmas_noise_sheffet)
    train_mse_for_sigmas_std_noise_sheffet = locals().get('train_mse_for_sigmas_std_noise_sheffet', mse_for_sigmas_std_noise_sheffet)
    train_mse_for_sigmas_adassp = locals().get('train_mse_for_sigmas_adassp', mse_for_sigmas_adassp)
    train_mse_for_sigmas_std_adassp = locals().get('train_mse_for_sigmas_std_adassp', mse_for_sigmas_std_adassp)
    train_mse_for_sigmas_noise_sheffet_our_analysis = locals().get('train_mse_for_sigmas_noise_sheffet_our_analysis', mse_for_sigmas_noise_sheffet_our_analysis)
    train_mse_for_sigmas_std_noise_sheffet_our_analysis = locals().get('train_mse_for_sigmas_std_noise_sheffet_our_analysis', mse_for_sigmas_std_noise_sheffet_our_analysis)
    train_mse_for_sigmas = locals().get('train_mse_for_sigmas', mse_for_sigmas)
    train_mse_for_sigmas_std = locals().get('train_mse_for_sigmas_std', mse_for_sigmas_std)

    # Left subplot: Train MSE
    ax1.plot(epsilon_values, train_mse_for_sigmas_noise_sheffet, 
             color=colors['sheffet'], marker=markers['sheffet'],
             label=fr"[Alg. 1, Sheffet '17]: {label_suffix}")
    ax1.fill_between(epsilon_values,
                     np.array(train_mse_for_sigmas_noise_sheffet) - np.array(train_mse_for_sigmas_std_noise_sheffet),
                     np.array(train_mse_for_sigmas_noise_sheffet) + np.array(train_mse_for_sigmas_std_noise_sheffet),
                     color=colors['sheffet'], alpha=0.2)

    ax1.plot(epsilon_values, train_mse_for_sigmas_adassp, 
             color=colors['adassp'], marker=markers['adassp'], markersize=12,
             label="ADASSP [Wang '18]")
    ax1.fill_between(epsilon_values,
                     np.array(train_mse_for_sigmas_adassp) - np.array(train_mse_for_sigmas_std_adassp),
                     np.array(train_mse_for_sigmas_adassp) + np.array(train_mse_for_sigmas_std_adassp),
                     color=colors['adassp'], alpha=0.2)

    ax1.plot(epsilon_values, train_mse_for_sigmas_noise_sheffet_our_analysis, 
             color=colors['sheffet_ours'], marker=markers['sheffet_ours'],
             label=fr"[Alg. 1, Sheffet '17] (ours): {label_suffix}")
    ax1.fill_between(epsilon_values,
                     np.array(train_mse_for_sigmas_noise_sheffet_our_analysis) - np.array(train_mse_for_sigmas_std_noise_sheffet_our_analysis),
                     np.array(train_mse_for_sigmas_noise_sheffet_our_analysis) + np.array(train_mse_for_sigmas_std_noise_sheffet_our_analysis),
                     color=colors['sheffet_ours'], alpha=0.2)

    ax1.plot(epsilon_values, train_mse_for_sigmas, 
             color=colors['linear_mixing'], marker=markers['linear_mixing'], markersize=12,
             label=fr"Linear mixing (ours): {label_suffix}")
    ax1.fill_between(epsilon_values,
                     np.array(train_mse_for_sigmas) - np.array(train_mse_for_sigmas_std),
                     np.array(train_mse_for_sigmas) + np.array(train_mse_for_sigmas_std),
                     color=colors['linear_mixing'], alpha=0.2)

    # Right subplot: Test MSE
    ax2.plot(epsilon_values, mse_for_sigmas_noise_sheffet, 
             color=colors['sheffet'], marker=markers['sheffet'],
             label=fr"[Alg. 1, Sheffet '17]: {label_suffix}")
    ax2.fill_between(epsilon_values,
                     np.array(mse_for_sigmas_noise_sheffet) - np.array(mse_for_sigmas_std_noise_sheffet),
                     np.array(mse_for_sigmas_noise_sheffet) + np.array(mse_for_sigmas_std_noise_sheffet),
                     color=colors['sheffet'], alpha=0.2)

    ax2.plot(epsilon_values, mse_for_sigmas_adassp, 
             color=colors['adassp'], marker=markers['adassp'], markersize=12,
             label="ADASSP [Wang '18]")
    ax2.fill_between(epsilon_values,
                     np.array(mse_for_sigmas_adassp) - np.array(mse_for_sigmas_std_adassp),
                     np.array(mse_for_sigmas_adassp) + np.array(mse_for_sigmas_std_adassp),
                     color=colors['adassp'], alpha=0.2)

    ax2.plot(epsilon_values, mse_for_sigmas_noise_sheffet_our_analysis, 
             color=colors['sheffet_ours'], marker=markers['sheffet_ours'],
             label=fr"[Alg. 1, Sheffet '17] (ours): {label_suffix}")
    ax2.fill_between(epsilon_values,
                     np.array(mse_for_sigmas_noise_sheffet_our_analysis) - np.array(mse_for_sigmas_std_noise_sheffet_our_analysis),
                     np.array(mse_for_sigmas_noise_sheffet_our_analysis) + np.array(mse_for_sigmas_std_noise_sheffet_our_analysis),
                     color=colors['sheffet_ours'], alpha=0.2)

    ax2.plot(epsilon_values, mse_for_sigmas, 
             color=colors['linear_mixing'], marker=markers['linear_mixing'], markersize=12,
             label=fr"Linear mixing (ours): {label_suffix}")
    ax2.fill_between(epsilon_values,
                     np.array(mse_for_sigmas) - np.array(mse_for_sigmas_std),
                     np.array(mse_for_sigmas) + np.array(mse_for_sigmas_std),
                     color=colors['linear_mixing'], alpha=0.2)

    # Add vertical dashed line at the predicted eps_th
    if add_vertical_line:
        ax1.axvline(x=np.sqrt(k_val)*np.sqrt(np.log(1/delta_DP))*2.0/baseline_train_mse, linestyle='--', color='green', 
           linewidth=3, alpha=0.8)
        ax2.axvline(x=np.sqrt(k_val)*np.sqrt(np.log(1/delta_DP))*2.0/baseline_train_mse, linestyle='--', color='green', 
           linewidth=3, alpha=0.8)
        
    # Configure both subplots
    for ax, ylabel in zip([ax1, ax2], ["Train MSE", "Test MSE"]):
        ax.set_xlabel(r"$\epsilon_{\mathrm{DP}}$", fontsize=26)
        ax.set_ylabel(ylabel, fontsize=26)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='both', labelsize=26)  # <-- not just major
        ax.grid(True)

        # DP delta annotation
        textstr = r"$\delta_{\mathrm{DP}} = \frac{1}{n^2}$"
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
                fontsize=24, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

    # Dataset-dependent suptitle
    dataset_titles = {
        'autompg'           : "Autompg"             ,
        'energy'            : "Energy"              ,
        'elevators'         : "Elevators"           ,
        'concrete'          : "Concrete"            ,
        'airfoil'           : "Airfoil"             ,
        'breastcancer'      : "Breast Cancer"       ,
        'parkinsons'        : "Parkinsons"          ,
        'sml'               : "SML"                 ,
        'keggundirected'    : "KEGG (Undirected)"   ,
        'challenger'        : "Challenger"          ,
        'protein'           : "Protein"             ,
        'crime'             : "Communities & Crime" ,
        'housing'           : "Boston Housing"      ,
        'bike'              : "Bike Sharing"        ,
        'wine'              : "Wine"                ,
        'tecator'           : "Tecator"             ,
        'Gas'               : "Gas"                 ,
        'Concrete'          : "Concrete"            ,
        'tamielectric'      : "Tami Electric"       ,
        'keggdirected'      : "KEGG (Directed)"     ,
        'yacht'             : "Yacht"               ,
        'solar'             : "Solar"               ,
        '3droad'            : "3D Road"             ,
        'slice'             : "Slice"               ,
        'servo'             : "Servo"               ,
        'autos'             : "Autos"               ,
        'concreteslump'     : "Concrete Slump"      ,
        'fertility'         : "Fertility"           ,
        'forest'            : "Forest"              ,
        'houseelectric'     : "House Electric"      ,
        'kin40k'            : "Kin40K"              ,
        'machine'           : "Machine"             ,
        'pol'               : "Pol"                 ,
        'pendulum'          : "Pendulum"            ,
        'pumadyn32nm'       : "Pumadyn32nm"         ,
        'buzz'              : "Buzz"                ,
        'synthetic_MLP'     : fr'Well-Specified Model: Low-rank X via MLP, $n = {n}$, $d = {d}$', # Synthetic dataset
        'synthetic_Gaussian': fr'Well-Specified Model: Low-rank X via normalized Gaussian, $n = {n}$, $d = {d}$',
        'synthetic_Uniform' : fr'Well-Specified Model: Low-rank X via Uniform samples, $n = {n}$, $d = {d}$',
        'synthetic_eig'     : fr'Well-Specified Model: X With Spherical Rows, $\frac{{n}}{{d}} = {n/d:.2f}$'
    }

    # Build a LaTeX-friendly suffix
    if lambda_min < 1e-4:
        lambda_min_plot = 0.0
    else:
        lambda_min_plot = lambda_min
    
    if lambda_min_XY < 1e-4:
        lambda_min_XY_plot = 0.0
    else:
        lambda_min_XY_plot = lambda_min_XY
        
    title_suffix = (
        rf"n={n}, d={d}, "
        rf"$\lambda_{{\min}}(X^{{\top}}X)={lambda_min_plot:.3f}$, "
        rf"$\lambda_{{\min}}^{{XY}}={lambda_min_XY_plot:.3f}$, "
        rf"$\frac{{\|R\|^2}}{{\gamma}}={R_over_gamma_plot:.3f}$, "
        rf"$\frac{{1}}{{n}}\|R\|^2 = {baseline_train_mse/n:.3f}$, "
        rf"$\frac{{1}}{{n}}\|Y\|^2 = {np.sum(y_train**2)/n:.3f}$"
    )

    # Two-line suptitle: dataset name on line 1, suffix on line 2
    fig.suptitle(
        f"{dataset_titles.get(dataset_type, '')}\n{title_suffix}",
        fontsize=26
    )

    # Legend (only one legend for the entire figure spanning below both plots)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=2, fontsize=24, frameon=False)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/linear_mixing_{dataset_type}_n_{n}_d_{d}_k_{percentage_k}.pdf", bbox_inches='tight')
