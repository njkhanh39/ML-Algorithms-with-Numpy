import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def gen_linear_2Ddata(n_train = 800, n_val = 100):
    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters
    n_train = n_train
    n_val = n_val

    # Generate full dataset (train + val) from the same population
    total_samples = n_train + n_val

    # Mean and covariance for the multivariate normal distribution
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]  # Independent standard Gaussians for x1 and x2

    # Generate features
    X = np.random.multivariate_normal(mean, cov, total_samples)

    # Generate target variable y = w^T x + noise
    true_weights = np.array([3.5, -2.0])
    noise = np.random.normal(0, 0.5, total_samples)  # Gaussian noise
    y = X @ true_weights + noise

    # Split into training and validation sets
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    return X_train, y_train, X_val, y_val
# in general
def load_data(train_dir = "data/log_reg_train.csv", valid_dir = "data/log_reg_valid.csv"):
    return pd.read_csv(train_dir).to_numpy(), pd.read_csv(valid_dir).to_numpy()

def extract_X_y(arr):
    X = arr[:,:-1]
    y = arr[:, -1] #1D array (last column) - shape (m,)
    return X,y

def plot_2D_data(X, y, theta, log_reg = True): #assume X.shape = (m, 2)
    plt.figure(figsize=(14,8))


    #data


    # X[y==0, 0] extract first column vector of X, whose entries corresponding to 
    # label 0
    if(log_reg == True):
        plt.plot(X[y==0, 0], X[y==0, 1], 'go', linewidth = 2)
        plt.plot(X[y==1, 0], X[y==1, 1], 'bx', linewidth = 2)
    else:
        plt.plot(X[:, 0], X[:, 1], 'go', linewidth = 2)


    
    # Plot decision boundary (found by solving for theta^T x = 0)
    margin1 = (max(X[:, 0]) - min(X[:, 0]))*0.2
    margin2 = (max(X[:, 1]) - min(X[:, 1]))*0.2
    x1 = np.arange(min(X[:, 0])-margin1, max(X[:, 0])+margin1, 0.01)
    x2 = -(theta[0] / theta[2] + theta[1] / theta[2] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)
    plt.xlim(X[:, 0].min()-margin1, X[:, 0].max()+margin1)
    plt.ylim(X[:, 1].min()-margin2, X[:, 1].max()+margin2)

    plt.xlabel('x1')
    plt.xlabel('x2')

def accuracy_score(x_pred, y_train):
    # Step 1: Convert probabilities to binary predictions (0 or 1)
    predictions = (x_pred >= 0.5).astype(int)  # threshold at 0.5

    #print(predictions.shape)
    
    # Step 2: Compare predictions with true labels
    correct_predictions = np.sum(predictions == y_train)

    # Step 3: Calculate accuracy
    accuracy = correct_predictions / len(y_train)
    
    return accuracy