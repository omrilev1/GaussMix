import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
import urllib 
import scipy
from scipy.special import expit
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo 

# Utility functions: find optimal noise for eps Renyi DP
# Define the function to minimize
def objective_func_full(alpha, k, d, n, sigma, delta,C_max):
    if alpha <= 1 or alpha >= sigma/(1+C_max):
        return np.inf  # Penalize out-of-bound values

    # term1 = (k * alpha) / (2 * (alpha - 1)) * np.log(1.0 - (1.0 + C_max) / (n*(1.0 + C_max) + sigma))
    term1 = (k * alpha) / (2 * (alpha - 1)) * np.log(1.0 - (1.0 + C_max) / (sigma))
    term2 = - (k / (2 * (alpha - 1))) * np.log(1 - (alpha*(1 + C_max)) / sigma)
    term3 = (np.log(3.0 / delta) + (alpha - 1)*np.log(1-1/alpha) - np.log(alpha)) / (alpha - 1)
    term4 = np.sqrt(2.0 * np.log(3.75/delta)) / (sigma/np.sqrt(k))
    return term1 + term2 + term3 + term4

# Solve with full conversion from RenyiDP to DP for the Linear mixing algorithm
def solve_sigma_renyi_full(sigma_DP, n_prime, d, n, delta, target_epsilon, C_max):   
    # Define binary search bounds
    left, right = sigma_DP / 500000.0, 500000.0*sigma_DP
    best_sigma = right  # Default to upper bound in case no solution is found
    while right - left > 1e-6:  # Precision threshold
        mid_sigma = (left + right) / 2
        # Solve for optimal alpha given the current sigma
        result = scipy.optimize.minimize_scalar(objective_func_full, 
                                 bounds=(1 + 1e-5, mid_sigma - 1e-5), 
                                 args=(n_prime, d, n, mid_sigma, delta,C_max), 
                                 method='bounded')
        if result.success and result.fun < target_epsilon:
            best_sigma = mid_sigma  # Update best found sigma
            right = mid_sigma  # Search for a smaller sigma
        else:
            left = mid_sigma  # Increase sigma to meet target_epsilon
    return best_sigma

# Define the function to minimize
def objective_func(alpha, k, d, n, sigma, delta,C_max):
    if alpha <= 1 or alpha >= sigma/(1+C_max):
        return np.inf  # Penalize out-of-bound values

    # term1 = (k * alpha) / (2 * (alpha - 1)) * np.log(1.0 - (1.0 + C_max) / (n*(1.0 + C_max) + sigma))
    term1 = (k * alpha) / (2 * (alpha - 1)) * np.log(1.0 - (1.0 + C_max) / (sigma))
    term2 = - (k / (2 * (alpha - 1))) * np.log(1 - (alpha*(1 + C_max)) / sigma)
    term3 = (np.log(1.0 / delta) + (alpha - 1)*np.log(1-1/alpha) - np.log(alpha)) / (alpha - 1)
    return term1 + term2 + term3

# Solve with full conversion from RenyiDP to DP with the classical conversion from 
# RenyiDP to DP 
def solve_sigma_renyi(sigma_DP, n_prime, d, n, delta, target_epsilon, C_max):   
    # Define binary search bounds
    left, right = sigma_DP / 30000.0, 30000.0*sigma_DP
    best_sigma = right  # Default to upper bound in case no solution is found
    while right - left > 1e-6:  # Precision threshold
        mid_sigma = (left + right) / 2
        # Solve for optimal alpha given the current sigma
        result = scipy.optimize.minimize_scalar(objective_func, 
                                 bounds=(1 + 1e-5, mid_sigma - 1e-5), 
                                 args=(n_prime, d, n, mid_sigma, delta,C_max), 
                                 method='bounded')
        if result.success and result.fun < target_epsilon:
            best_sigma = mid_sigma  # Update best found sigma
            right = mid_sigma  # Search for a smaller sigma
        else:
            left = mid_sigma  # Increase sigma to meet target_epsilon
    return best_sigma

# Load crimes dataset
def read_crimes(label='ViolentCrimesPerPop', sensitive_attribute='racepctblack', env_partition=0.05):
    if not os.path.isfile('communities.data'):
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data", "communities.data")
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names",
            "communities.names")
    # create names
    names = []
    with open('communities.names', 'r') as file:
        for line in file:
            if line.startswith('@attribute'):
                names.append(line.split(' ')[1])
    # load data
    data = pd.read_csv('communities.data', names=names, na_values=['?'])
    data.drop(['state', 'county', 'community', 'fold', 'communityname'], axis=1, inplace=True)
    data = data.replace('?', np.nan)
    data['OtherPerCap'] = data['OtherPerCap'].fillna(data['OtherPerCap'].astype(float).mean())
    data = data.dropna(axis=1)
    data['OtherPerCap'] = data['OtherPerCap'].astype(float)
    # shuffle
    data = data.sample(frac=1, replace=False).reset_index(drop=True)
    to_drop = []
    y = data[label].values
    to_drop += [label]
    z = data[sensitive_attribute].values
    to_drop += [sensitive_attribute]
    data.drop(to_drop + [label], axis=1, inplace=True)
    for n in data.columns:
        data[n] = (data[n] - data[n].mean()) / data[n].std()
    x = np.array(data.values)
    x = x[z >= env_partition]
    y = y[z >= env_partition]
    z = z[z >= env_partition]
    
    # Apply random permutation 
    m, n = x.shape
    p = np.random.permutation(m)
    x = x[p, :]
    y = y[p]
    
    # Split into training and testing sets
    train_size = int(0.8 * len(y))

    X_train = x[:train_size]
    y_train = y[:train_size]
    X_test = x[train_size:]
    y_test = y[train_size:]
    train_size = int(0.8 * len(y))

    norm_fact_y = np.max(np.abs(y_train))
    y_train /= norm_fact_y
    y_test  /= norm_fact_y
        
    return X_train,X_test,z,z,y_train,y_test
    # return train_test_split(x, E, y, test_size=0.2, random_state = 0)

# Load datasets 
def GetDataset(dataset_name):
    
    # current directory of datasets files 
    current_dir = os.getcwd()
    dataset_path = os.path.join(current_dir, "datasets")
    
    # Boston housing 
    if dataset_name == 'housing':

        data = np.loadtxt(os.path.join(dataset_path, 'housing.txt'))

        # 2. Shuffle the rows (similar to p = randperm(m)).
        m, n = data.shape
        p = np.random.permutation(m)
        data = data[p, :]

        # 3. Separate features (X) and target (y).
        X = data[:, :-1]
        y = data[:, -1]

        # apply train test split 
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0

    # Elevators 
    elif dataset_name == 'elevators':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'elevators_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'elevators_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # sml 
    elif dataset_name == 'sml':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'sml_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'sml_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # protein 
    elif dataset_name == 'protein':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'protein_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'protein_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # keggle undirected 
    elif dataset_name == 'keggundirected':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'keggundirected_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'keggundirected_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # keggle directed 
    elif dataset_name == 'keggdirected':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'keggdirected_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'keggdirected_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0

    # 3droad
    elif dataset_name == '3droad':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, '3droad_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, '3droad_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # slice
    elif dataset_name == 'slice':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'slice_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'slice_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # servo
    elif dataset_name == 'servo':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'servo_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'servo_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # autos
    elif dataset_name == 'autos':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'autos_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'autos_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # concreteslump
    elif dataset_name == 'concreteslump':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'concreteslump_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'concreteslump_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # fertility
    elif dataset_name == 'fertility':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'fertility_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'fertility_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # forest
    elif dataset_name == 'forest':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'forest_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'forest_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # houseelectric
    elif dataset_name == 'houseelectric':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'houseelectric_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'houseelectric_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
        
    # yacht
    elif dataset_name == 'yacht':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'yacht_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'yacht_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
        
    # kin40k
    elif dataset_name == 'kin40k':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'kin40k_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'kin40k_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # machine
    elif dataset_name == 'machine':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'machine_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'machine_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
        
    # solar
    elif dataset_name == 'solar':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'solar_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'solar_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # pol
    elif dataset_name == 'pol':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'pol_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'pol_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # pendulum
    elif dataset_name == 'pendulum':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'pendulum_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'pendulum_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # pumadyn32nm
    elif dataset_name == 'pumadyn32nm':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'pumadyn32nm_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'pumadyn32nm_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # buzz
    elif dataset_name == 'buzz':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'buzz_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'buzz_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
                                   
    # Breast Cancer dataset
    elif dataset_name == 'breastcancer':
        
        # Load dataset 
        data = np.loadtxt(os.path.join(dataset_path, 'breastcancer.txt'), delimiter=",")
        
        # Preprocess dataset 
        m, n = data.shape

        # Randomly permute the rows
        p = np.random.permutation(m)
        data = data[p, :]

        # Select columns [1:2, 4:end-1] (0-indexed: [0, 1] and [3:n-1])
        X = data[:, list(range(0, 2)) + list(range(3, n - 1))]
        y = data[:, 2]
        
        # apply train test split 
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0
        
    # autompg 
    elif dataset_name == 'autompg':
        # fetch dataset 
        auto_mpg = fetch_ucirepo(id=9) 
  
        # data (as pandas dataframes) 
        X = auto_mpg.data.features 
        y = auto_mpg.data.targets 
        X = X.to_numpy()
        y = y.to_numpy().ravel()
        
        # Delete missing values 
        data = np.concatenate([X, y[:, np.newaxis]], axis=1)

        # Remove rows with any NaNs
        data = data[~np.isnan(data).any(axis=1)]

        # Separate back into X and y
        X = data[:, :-1]
        y = data[:, -1]

        # 2. Shuffle the rows (similar to p = randperm(m)).
        m, n = X.shape
        p = np.random.permutation(m)
        data = data[p, :]

        # 3. Separate features (X) and target (y).
        X = data[:, :-1]
        y = data[:, -1]

        # apply train test split 
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0
    
    # challenger 
    elif dataset_name == 'challenger':
        # chellenger dataset 
        challenger = pd.read_csv(os.path.join(dataset_path, "ChallengerDataset.csv"))

        # Convert to NumPy once
        data = challenger.to_numpy().astype(np.float64)
    
        # Split into features (X) and target (y)
        X = data[:, :-1]
        y = data[:, -1]
        
        # Apply random permutation and apply train test split
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        y = y[p]

        X_train = X
        y_train = y
        X_test = X
        y_test = y
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max       = 1.0
    
    # parkinsons 
    elif dataset_name == 'parkinsons':
        # fetch dataset 
        parkinsons = fetch_ucirepo(id=174) 
  
        # data (as pandas dataframes) 
        X = parkinsons.data.features 
        y = parkinsons.data.targets 
        X = X.to_numpy().astype(np.float64)
        y = y.to_numpy().ravel().astype(np.float64)
        
        # Delete missing values 
        data = np.concatenate([X, y[:, np.newaxis]], axis=1)

        # Remove rows with any NaNs
        data = data[~np.isnan(data).any(axis=1)]

        # Separate back into X and y
        X = data[:, :-1]
        y = data[:, -1]

        # 2. Shuffle the rows (similar to p = randperm(m)).
        m, n = X.shape
        p = np.random.permutation(m)
        data = data[p, :]

        # 3. Separate features (X) and target (y).
        X = data[:, :-1]
        y = data[:, -1]

        # apply train test split 
        X_train = X
        y_train = y
        X_test = X
        y_test = y
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0
        
    # airfoil 
    elif dataset_name == 'airfoil':
        # fetch dataset 
        airfoil_self_noise = fetch_ucirepo(id=291) 
  
        # data (as pandas dataframes) 
        X = airfoil_self_noise.data.features 
        y = airfoil_self_noise.data.targets 
        X = X.to_numpy()
        y = y.to_numpy().ravel()
        
        # Delete missing values 
        data = np.concatenate([X, y[:, np.newaxis]], axis=1)

        # Remove rows with any NaNs
        data = data[~np.isnan(data).any(axis=1)]

        # Separate back into X and y
        X = data[:, :-1]
        y = data[:, -1]

        # 2. Shuffle the rows (similar to p = randperm(m)).
        m, n = X.shape
        p = np.random.permutation(m)
        data = data[p, :]

        # 3. Separate features (X) and target (y).
        X = data[:, :-1]
        y = data[:, -1]

        # apply train test split 
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0
        
    # concrete dataset 
    elif dataset_name == 'concrete':
        # fetch dataset 
        concrete_compressive_strength = fetch_ucirepo(id=165) 
  
        # data (as pandas dataframes) 
        X = concrete_compressive_strength.data.features 
        y = concrete_compressive_strength.data.targets 
        X = X.to_numpy()
        y = y.to_numpy().ravel()
        
        # Delete missing values 
        data = np.concatenate([X, y[:, np.newaxis]], axis=1)

        # Remove rows with any NaNs
        data = data[~np.isnan(data).any(axis=1)]

        # Separate back into X and y
        X = data[:, :-1]
        y = data[:, -1]

        # 2. Shuffle the rows (similar to p = randperm(m)).
        m, n = X.shape
        p = np.random.permutation(m)
        data = data[p, :]

        # 3. Separate features (X) and target (y).
        X = data[:, :-1]
        y = data[:, -1]

        # apply train test split 
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0
        
    # Energy dataset 
    elif dataset_name == 'energy':
        # fetch dataset 
        energy_efficiency = fetch_ucirepo(id=242) 
  
        # data (as pandas dataframes) 
        X = energy_efficiency.data.features 
        y = energy_efficiency.data.targets  # Ensure y matches X length
        X = X.to_numpy()
        y = y.to_numpy()[:, 0].ravel()
        
        # Delete missing values 
        data = np.concatenate([X, y[:, np.newaxis]], axis=1)

        # Remove rows with any NaNs
        data = data[~np.isnan(data).any(axis=1)]

        # Separate back into X and y
        X = data[:, :-1]
        y = data[:, -1]

        # 2. Shuffle the rows (similar to p = randperm(m)).
        m, n = X.shape
        p = np.random.permutation(m)
        data = data[p, :]

        # 3. Separate features (X) and target (y).
        X = data[:, :-1]
        y = data[:, -1]

        # apply train test split 
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0
        
    # Bike sharing 
    elif dataset_name == 'bike':
        # https://www.kaggle.com/rajmehra03/bike-sharing-demand-rmsle-0-3194
        df=pd.read_csv(os.path.join(dataset_path, 'bike_train.csv'))

        # # seperating season as per values. this is bcoz this will enhance features.
        season=pd.get_dummies(df['season'],prefix='season')
        df=pd.concat([df,season],axis=1)

        # # # same for weather. this is bcoz this will enhance features.
        weather=pd.get_dummies(df['weather'],prefix='weather')
        df=pd.concat([df,weather],axis=1)

        # # # now can drop weather and season.
        df.drop(['season','weather'],inplace=True,axis=1)
        df.head()

        df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]
        df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]
        df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = [t.year for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = df['year'].map({2011:0, 2012:1})
    
        df.drop('datetime',axis=1,inplace=True)
        df.drop(['casual','registered'],axis=1,inplace=True)
        df.columns.to_series().groupby(df.dtypes).groups
        X = df.drop('count',axis=1).values.astype(np.float64)
        y = df['count'].values.astype(np.float64)

        # Apply random permutation 
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        y = y[p]
        
        # Split into training and testing sets
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0 
    
    # Tecator 
    elif dataset_name == 'tecator':
        tecator = fetch_openml(name="Tecator", version=1, as_frame=True)

        # Extract the 100 absorbance features (cols are named like "Absorbance1", …)
        X = tecator.data.to_numpy()

        # Choose "Fat" as the regression target (you could also predict Moisture or Protein)
        y = tecator.target.astype(float).to_numpy()

        # Apply random permutation and apply train test split
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        y = y[p]
        
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]

        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0
    
    # Wine dataset
    elif dataset_name == 'wine':
        wine = pd.read_csv(os.path.join(dataset_path, "winequality-red.csv"))

        # Drop duplicates 
        dub_wine=wine.copy()
        dub_wine.drop_duplicates(subset=None,inplace=True)

        y=dub_wine.pop('quality').to_numpy().astype(np.float64)
        X=dub_wine.to_numpy().astype(np.float64)
        
        # Apply random permutation and apply train test split
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        y = y[p]
        
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max       = 1.0

    # SkillCraft dataset 
    elif dataset_name == 'SkillCraft':
        skillcraft = pd.read_csv(os.path.join(dataset_path, "SkillCraft.csv"))

        # Drop duplicates 
        dub_skillcraft=skillcraft.copy()
        dub_skillcraft.dropna(subset=None,inplace=True)

        y = dub_skillcraft['Age'].to_numpy().astype(np.float64)
        X = dub_skillcraft.drop(columns=['GameID', 'Age']).to_numpy()
        
        # Apply random permutation and apply train test split
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        y = y[p]
        
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max       = 1.0
    
    # Gas dataset 
    elif dataset_name == 'Gas':
        gas = pd.read_csv(os.path.join(dataset_path, "GasDataset.csv"))

        # Drop duplicate rows and missing values
        gas_cleaned = gas.drop_duplicates().dropna()

        # Convert to NumPy once
        data = gas_cleaned.to_numpy().astype(np.float64)
    
        # Split into features (X) and target (y)
        X = data[:, :-1]
        y = data[:, -1]
        
        # Apply random permutation and apply train test split
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        y = y[p]
        
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max       = 1.0
        
    # Synthetic dataset with uniform covariates matrix 
    elif dataset_name == 'synthetic_Uniform':
        n_samples = 2**13 + 2**11 
        n_features = 2**10 

        w_star = np.random.randn(n_features)
        w_star /= np.linalg.norm(w_star)  # Normalize to have ||w*||=1

        P = np.random.normal(size=(n_features, int(n_features/10)))  # Random matrix
        Q, _ = scipy.linalg.qr(P, mode='economic')  # QR decomposition to ensure orthogonality
        P = Q[:, :int(n_features/120)]  # Keep only the first `rank` orthogonal columns
        Sigma = P @ P.T  # Sigma will be (d x d), symmetric, and rank-deficient

        X = ((np.random.rand(n_samples, int(n_features/120))) * 2.0 - 1.0) * np.sqrt(12)
        X = X @ P.T
        norms = np.linalg.norm(X, axis=1)

        sigma_z = 0.1
        Z_tot = np.random.uniform(low=-sigma_z, high=sigma_z, size=n_samples)
        Y_tot = X @ w_star + Z_tot

        Y_tot[np.abs(Y_tot) > 1.0] == 1.0 * np.sign(Y_tot[np.abs(Y_tot) > 1.0])

        # Apply random permutation and apply train test split
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        Y_tot = Y_tot[p]
        
        train_size = int(0.8 * n_samples)
        test_size = n_samples - train_size

        X_train = X[:train_size]
        y_train = Y_tot[:train_size]
        X_test = X[train_size:]
        y_test = Y_tot[train_size:]
        C_max = 1.0 
        
    # Synthetic Minimum eigenvalue dataset 
    # Synthetic dataset with rows sampled uniformly from the d-dimensional unit sphere
    elif dataset_name == 'synthetic_eig':
        # number of samples and features
        n_samples = 2**13 + 2**13
        n_features = 2**5

        # Define w_star as a random unit vector
        w_star = np.random.randn(n_features)
        w_star /= np.linalg.norm(w_star)  # ||w*|| = 1

        # Generate X: rows uniformly from the unit sphere in R^d
        X = np.random.randn(n_samples, n_features)
        X /= np.linalg.norm(X, axis=1, keepdims=True)  # Normalize each row

        # Generate output with bounded uniform noise
        sigma_z = 0.01
        Z_tot = np.random.uniform(low=-sigma_z, high=sigma_z, size=n_samples)
        Y_tot = X @ w_star + Z_tot

        # Clip |Y| > 1 to be sign(Y)
        Y_tot[np.abs(Y_tot) > 1.0] = 1.0 * np.sign(Y_tot[np.abs(Y_tot) > 1.0])
        C_max = 1.0

        # Shuffle and split into train/test sets
        p = np.random.permutation(n_samples)
        X = X[p, :]
        Y_tot = Y_tot[p]

        train_size = int(0.5 * n_samples)
        test_size = n_samples - train_size

        X_train = X[:train_size]
        y_train = Y_tot[:train_size]
        X_test = X[train_size:]
        y_test = Y_tot[train_size:]

    # Synthetic Minimum eigenvalue dataset 
    # Synthetic dataset with rows sampled uniformly from the d-dimensional unit sphere
    elif dataset_name == 'synthetic_eig_high_noise':
        # number of samples and features
        n_samples = 2**12 + 2**12
        n_features = 2**4

        # Define w_star as a random unit vector
        w_star = np.random.randn(n_features)
        w_star /= np.linalg.norm(w_star)  # ||w*|| = 1

        # Generate X: rows uniformly from the unit sphere in R^d
        X = np.random.randn(n_samples, n_features)
        X /= np.sqrt(5)
        # X /= np.linalg.norm(X, axis=1, keepdims=True)  # Normalize each row

        # Generate output with bounded uniform noise
        sigma_z = 1.25
        Z_tot = np.random.uniform(low=-sigma_z, high=sigma_z, size=n_samples)
        Y_tot = X @ w_star + Z_tot

        # Clip |Y| > 1 to be sign(Y)
        Y_tot[np.abs(Y_tot) > 1.0] = 1.0 * np.sign(Y_tot[np.abs(Y_tot) > 1.0])
        C_max = 1.0

        # Shuffle and split into train/test sets
        p = np.random.permutation(n_samples)
        X = X[p, :]
        Y_tot = Y_tot[p]

        train_size = int(0.5 * n_samples)
        test_size = n_samples - train_size

        X_train = X[:train_size]
        y_train = Y_tot[:train_size]
        X_test = X[train_size:]
        y_test = Y_tot[train_size:]
        
    # Synthetic dataset such that the rows are sampled with one coordinate much weaker than others
    elif dataset_name == 'synthetic_diag_Gauss':
        # Number of samples and features
        n_samples = 2**12 + 2**12
        n_features = 2**6

        # Define anisotropic covariance: diag(1, ..., 1, eps)
        eps = 1e-5
        Sigma_diag = np.ones(n_features)/(4.0 * np.sqrt(n_features))
        Sigma_diag[-int(n_features/8):] = eps  # Last coordinates has much lower variance
        Sigma = np.diag(Sigma_diag)

        # Define w_star as a random unit vector
        w_star = np.random.randn(n_features)
        w_star /= np.linalg.norm(w_star)

        # Sample rows from N(0, Sigma)
        X = np.random.multivariate_normal(mean=np.zeros(n_features), cov=Sigma, size=n_samples)

        # Add bounded uniform noise
        sigma_z = 0.00001
        Z_tot = np.random.uniform(low=-sigma_z, high=sigma_z, size=n_samples)
        Y_tot = X @ w_star + Z_tot

        # Clip |Y| > 1 to ±1
        Y_tot[np.abs(Y_tot) > 1.0] = 1.0 * np.sign(Y_tot[np.abs(Y_tot) > 1.0])

        # Constant for bounds (if needed)
        C_max = 1.0

        # Shuffle and split into train/test sets
        p = np.random.permutation(n_samples)
        X = X[p, :]
        Y_tot = Y_tot[p]

        train_size = int(0.5 * n_samples)
        test_size = n_samples - train_size

        X_train = X[:train_size]
        y_train = Y_tot[:train_size]
        X_test = X[train_size:]
        y_test = Y_tot[train_size:]

    # Synthetic Gaussian dataset
    elif dataset_name == 'synthetic_Gaussian':
        # number of samples and features
        n_samples = 2**13 + 2**11  # 2**13 + 2**11
        n_features = 2**10         # 2**9

        w_star = np.random.randn(n_features)
        w_star /= np.linalg.norm(w_star)  # Normalize to have ||w*||=1

        P = np.random.normal(size=(n_features, int(n_features/10)))  # Random matrix
        Q, _ = scipy.linalg.qr(P, mode='economic')  # QR decomposition to ensure orthogonality
        P = Q[:, :int(n_features/120)]  # Keep only the first `rank` orthogonal columns
        Sigma = P @ P.T  # Sigma will be (d x d), symmetric, and rank-deficient

        X = np.random.multivariate_normal(mean=np.zeros(n_features), cov=Sigma, size=n_samples)
        norms = np.linalg.norm(X, axis=1)

        sigma_z = 0.1
        Z_tot = np.random.uniform(low=-sigma_z, high=sigma_z, size=n_samples)
        Y_tot = X @ w_star + Z_tot

        Y_tot[np.abs(Y_tot) > 1.0] == 1.0 * np.sign(Y_tot[np.abs(Y_tot) > 1.0])
        C_max = 1.0

        # Apply random permutation and apply train test split
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        Y_tot = Y_tot[p]
        train_size = int(0.8 * n_samples)
        test_size = n_samples - train_size

        X_train = X[:train_size]
        y_train = Y_tot[:train_size]
        X_test = X[train_size:]
        y_test = Y_tot[train_size:]

    # Synthetic Full-Rank Gaussian dataset
    elif dataset_name == 'synthetic_FullRank_Gaussian':
        # number of samples and features
        n_samples = 2**13 + 2**11
        n_features = 2**9 

        w_star = np.random.randn(n_features)
        w_star /= np.linalg.norm(w_star)  # Normalize to have ||w*||=1

        X = np.random.multivariate_normal(mean=np.zeros(n_features), cov= (2.0/n_features)* np.eye(n_features), size=n_samples)
        norms = np.linalg.norm(X, axis=1)

        sigma_z = 0.1
        Z_tot = np.random.uniform(low=-sigma_z, high=sigma_z, size=n_samples)
        Y_tot = X @ w_star + Z_tot

        Y_tot[np.abs(Y_tot) > 1.0] == 1.0 * np.sign(Y_tot[np.abs(Y_tot) > 1.0])
        C_max = 1.0

        # Apply random permutation and apply train test split
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        Y_tot = Y_tot[p]
        train_size = int(0.8 * n_samples)
        test_size = n_samples - train_size

        X_train = X[:train_size]
        y_train = Y_tot[:train_size]
        X_test = X[train_size:]
        y_test = Y_tot[train_size:]
        
    # Synthetic low rank 
    # Synthetic Gaussian dataset
    elif dataset_name == 'synthetic_low_rank':
        # number of samples and features
        n_samples = 2**13 + 2**11
        n_features = 2**8

        w_star = np.random.randn(n_features)
        w_star /= np.linalg.norm(w_star)  # Normalize to have ||w*||=1
        
        # Initialize low-rank design matrix
        X = np.zeros((n_samples, n_features))

        # Add 4 rank-1 components
        for _ in range(1):
            col_vec = np.random.randn(n_samples, 1)
            row_vec = np.random.randn(1, n_features)

            # Normalize row_vec for stability (optional)
            row_vec /= np.linalg.norm(row_vec)

            X += col_vec @ row_vec  # Outer product and accumulate
    
        sigma_z = 0.5
        Z_tot = np.random.uniform(low=-sigma_z, high=sigma_z, size=n_samples)
        Y_tot = X @ w_star + Z_tot

        Y_tot[np.abs(Y_tot) > 1.0] == 1.0 * np.sign(Y_tot[np.abs(Y_tot) > 1.0])
        C_max = 1.0

        # Apply random permutation and apply train test split
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        Y_tot = Y_tot[p]
        train_size = int(0.8 * n_samples)
        test_size = n_samples - train_size

        X_train = X[:train_size]
        y_train = Y_tot[:train_size]
        X_test = X[train_size:]
        y_test = Y_tot[train_size:]

    elif dataset_name == 'crime':
        # load crime dataset, standardize features, and split into train and test
        X_train, X_test, z_train, z_test, y_train, y_test = read_crimes(env_partition=0.1)
        test_size = len(y_test)
        n_samples = len(y_train) + test_size

        C_max = 1.0 

    # Tamielectric dataset
    elif dataset_name == 'tamielectric':
        # load the tamielectric dataset 
        tamielectric = pd.read_csv(os.path.join(dataset_path, "tamielectric.csv"))
        tamielectric_numpy = tamielectric.to_numpy().astype(np.float64)
        
        # Drop duplicates 
        y=tamielectric_numpy[:, -1]
        X=tamielectric_numpy[:, :-1]
        
        # Apply random permutation and apply train test split
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        y = y[p]
        
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe
        
        C_max       = 1.0
        
    # Synthetic dataset with MLP features and continuous responses 
    elif dataset_name == 'synthetic_MLP':
        # Set parameters
        n_samples = 2**13 + 2**11
        n_features = 2
        layer_sizes = [2**10, 2**10, 2**10] # [2**9, 2**9, 2**9] # [100, 100, 100]
        mu, sigma = 1, 1  # For lognormal distribution

        X = np.random.lognormal(mean=mu, sigma=sigma, size=(n_samples, n_features))

        # Initialize weights and biases for the MLP
        theta = np.random.normal(size=(layer_sizes[2], 1))
        weights = [
            np.random.normal(size=(n_features, layer_sizes[0])),
            np.random.normal(size=(layer_sizes[0], layer_sizes[1])),
            np.random.normal(size=(layer_sizes[1], layer_sizes[2])),
            theta/np.sqrt(np.sum(theta**2))
        ]

        biases = [
            0.001 * np.random.normal(size=(size,)) for size in layer_sizes + [1]
        ]

        # Generate the response variable
        def mlp_forward(X, weights, biases):
            layer_output = X
            for w, b in zip(weights[:-1], biases[:-1]):
                layer_output = expit(np.dot(layer_output, w) + b)
            # Output layer
            output = np.dot(layer_output, weights[-1]) + biases[-1]
            return output, layer_output

        output, mid_X = mlp_forward(X, weights, biases)
        sigma_z = 0.1 
        Y = output.flatten() + sigma_z * np.random.normal(size=n_samples)
        Y[np.abs(Y) > 1.0] == 1.0 * np.sign(Y[np.abs(Y) > 1.0])
        C_max = 1.0

        # Apply random permutation and apply train test split
        m, n = mid_X.shape
        p = np.random.permutation(m)
        mid_X = mid_X[p, :]
        Y = Y[p]

        train_size = int(0.8 * n_samples)
        test_size = n_samples - train_size
        X_train = mid_X[:train_size]
        y_train = Y[:train_size]
        X_test = mid_X[train_size:]
        y_test = Y[train_size:]
    
    # Synthetic dataset with MLP features and continuous responses 
    elif dataset_name == 'synthetic_MLP_finite_residual':
        # Set parameters
        n_samples = 2**11 + 2**11
        n_features = 2
        layer_sizes = [2**11, 2**11, 2**11] # [2**9, 2**9, 2**9] # [100, 100, 100]
        mu, sigma = 1, 1  # For lognormal distribution

        X = np.random.lognormal(mean=mu, sigma=sigma, size=(n_samples, n_features))

        # Initialize weights and biases for the MLP
        theta = np.random.normal(size=(layer_sizes[2], 1))
        weights = [
            np.random.normal(size=(n_features, layer_sizes[0])),
            np.random.normal(size=(layer_sizes[0], layer_sizes[1])),
            np.random.normal(size=(layer_sizes[1], layer_sizes[2])),
            theta/np.sqrt(np.sum(theta**2))
        ]

        biases = [
            0.001 * np.random.normal(size=(size,)) for size in layer_sizes + [1]
        ]

        # Generate the response variable
        def mlp_forward(X, weights, biases):
            layer_output = X
            for w, b in zip(weights[:-1], biases[:-1]):
                layer_output = expit(np.dot(layer_output, w) + b)
            # Output layer
            output = np.dot(layer_output, weights[-1]) + biases[-1]
            return output, layer_output

        output, mid_X = mlp_forward(X, weights, biases)
        sigma_z = 1/np.sqrt(n_samples) 
        Y = output.flatten() + sigma_z * np.random.normal(size=n_samples)
        Y[np.abs(Y) > 1.0] == 1.0 * np.sign(Y[np.abs(Y) > 1.0])
        C_max = 1.0

        # Apply random permutation and apply train test split
        m, n = mid_X.shape
        p = np.random.permutation(m)
        mid_X = mid_X[p, :]
        Y = Y[p]

        train_size = int(0.8 * n_samples)
        test_size = n_samples - train_size
        X_train = mid_X[:train_size]
        y_train = Y[:train_size]
        X_test = mid_X[train_size:]
        y_test = Y[train_size:]
        
    # normalize dataset to have power 1 
    norm_fact = np.sqrt(np.max(np.sum(X_train**2, 1)))  # np.mean(np.mean(X_train**2))
    X_train = X_train/norm_fact
    X_test = X_test/norm_fact
    n, d = X_train.shape

    # Calculate minimum eigenvalue of X^T X and of (X,Y)^T (X,Y)
    lambda_min = np.min(np.linalg.eigvals(X_train.T @ X_train))
    y_col = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
    XY = np.hstack((X_train, y_col))  
    lambda_min_XY = np.real(np.min(np.linalg.eigvals(XY.T @ XY)))
    
    return C_max, X_train, y_train, X_test, y_test, np.real(lambda_min), np.real(lambda_min_XY), n, d
