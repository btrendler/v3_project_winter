"""
Import Statements
"""

import rescomp as rc
import numpy as np
from scipy.interpolate import CubicSpline
from scipy import integrate, sparse
import math 
import networkx as nx
import itertools
import csv
import time
import pyedflib
from pyedflib import highlevel
from scipy.signal import detrend
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [20, 5]

# Set seed for reproducibility
np.random.seed(1)

# Diversity Metric Imports
from math import comb



"""
Reservoir Computing Helper Functions
"""

def train_and_drive(A, n, gamma, rho, sigma, W_in, u, U_train, t, T=200):
    """ Train and Drive the reservoir computer
    """

    print(f"train and drive t shape: {t.shape}")
    
    # ODE IVP definition and numerical solution
    drdt = lambda rs, ts : gamma * (-rs + np.tanh(rho * A @ rs + sigma * W_in @ u(ts)))
    r0 = np.random.rand(n)

    print(np.tanh(rho*A @ r0 + sigma * W_in @ u(0)))
    print("Pre states")
    states = integrate.odeint(drdt, r0, t[:T], printmessg=True)

    print(f"Post states")

    # Training step. Project training data onto reservoir states 
    W_out =  U_train.T @ states @ np.linalg.inv(states.T @ states )

    print(f"Post W_out")

    # Prediction ODE IVP definition and solution
    trained_drdt = lambda r, t : gamma * (-r + np.tanh(rho * A @ r + sigma * W_in @ W_out @ r))
    r0_pred = states[-1, :]
    pred_states = integrate.odeint(trained_drdt, r0_pred, t[T:])

    print("Post pred_states")

    # Map reservoir states onto the dynamical system space
    U_pred = W_out @ pred_states.T

    return pred_states, U_pred


"""
Automating Tests
"""

def div_metric_tests(preds, T, n):
    """ Compute Diversity scores of predictions
    """
    # Take the derivative of the pred_states
    res_deriv = np.gradient(preds[:T], axis=0)

    # Run the metric for the old and new diversity scores
    div = 0
    old_div = 0
    for i in range(n):
        for j in range(n):
            div += np.sum(np.abs(np.abs(preds[:T, i]) - np.abs(preds[:T, j])))
            old_div += np.sum(np.abs(res_deriv[:T, i] - res_deriv[:T, j]))
    div = div / (T*comb(n,2))
    old_div = old_div / (T*comb(n,2))

    return div, old_div

def remove_edges(A,n_edges):
    """ Randomly removes 'n_edges' edges from a sparse matrix 'A'
    """
    B = A.copy().todok() # - - - - - - - -  set A as copy

    keys = list(B.keys()) # - - - - remove edges
   
    remove_idx = np.random.choice(range(len(keys)),size=n_edges, replace=False)
    remove = [keys[i] for i in remove_idx]
    for e in remove:
        B[e] = 0
    return B



"""
Gridsearch Parameter Setup
"""

def gridsearch_dict_setup():
    # Topological Parameters
    ns = [50, 100, 150]
    p_thins = [0.2, 0.4, 0.8]

    # Model Specific Parameters
    erdos_renyi_c = [3,4]
    random_digraph_c = [.5,1,2,3,4]
    random_geometric_c = [.5,1,2,3,4]
    barabasi_albert_m = [1,2]
    watts_strogatz_k = [2,4]
    watt_strogatz_q = [.01,.05,.1]

    # Reservoir Computing Parameters
    gammas = [1,2,10,50]
    rhos = [1,5,10,50]
    sigmas = [1e-2,.14,10]
    alphas = [1e-2]

    erdos_possible_combinations = list(itertools.product(ns, p_thins, erdos_renyi_c, gammas, rhos, sigmas, alphas))
    # digraph_possible_combinations = list(itertools.product(ns, p_thins, random_digraph_c, gammas, rhos, sigmas, alphas))
    # geometric_possible_combinations = list(itertools.product(ns, p_thins, random_geometric_c, gammas, rhos, sigmas, alphas))
    # barabasi_possible_combinations = list(itertools.product(ns, p_thins, barabasi_albert_m, gammas, rhos, sigmas, alphas))
    # strogatz_possible_combinations = list(itertools.product(ns, p_thins, watts_strogatz_k, watt_strogatz_q, gammas, rhos, sigmas, alphas))

    return erdos_possible_combinations



"""
Training Data
"""

def get_patient_data(load_psg=False):
    """ Get Patient Data for Reservoir Computing
    """
    print("Start Loading data")

    if load_psg:
        signals, signal_headers, header = highlevel.read_edf('../SC4011E0-PSG.edf')
        signal = np.array(signals[:3])

        window_size = 4
        sleep_start = 2_200_000
        sleep_end = 2_200_250
        
        # Convert array of integers to pandas series
        signal_dataframe = pd.DataFrame((signal[:,sleep_start:sleep_end]).T, columns=['0', '1', '2'])

        # Get the window of series
        # of observations of specified window size and compute mean
        moving_averages = signal_dataframe.rolling(window_size, min_periods=1, axis=0).mean()
        moving_averages.dropna(inplace=True)

        # Take the moving average to mask out system noise
        U = moving_averages.values
        n = len(U[:,0])

        # Training Data
        t = np.linspace(sleep_start, sleep_end, n)
        return t, U
    else:
        data_path = 'sc-agg-f16.npz'
        data = np.load(data_path)

        patient_name = 'SC4111'
        patient = data[patient_name]

        # Number of time steps, number of columns
        T, C = patient.shape

        # Analyze just the sleep time
        sleep_start = np.argmax(patient[:, -1] > 0)
        sleep_end = T - np.argmax(patient[:, -1][::-1] > 0)
        patient_sleep = patient[sleep_start:sleep_end]
        n = len(patient_sleep[:, -1])

        # Training Data
        t = np.linspace(sleep_start, sleep_end, n)[:250]

        # Time Series with first frequencies up to the first 300 iterations
        U = patient_sleep[:250,:2]
        print("Finish loading data")
        return t, U



"""
Perform the Gridsearch
"""

def run_gridsearch(erdos_possible_combinations, t, U, T=200, iterations=30, tf=170_000):
    """ Run the gridsearch over possible combinations
    """

    # Interpolate data
    u = CubicSpline(t, U)
    _, dim = U.shape
    U_train = u(t[:T])
    # print(f"dimension: {dim}")

    # Parameters
    epsilon = 5
    test_t = t[T:]
    t0 = time.time()

    mse_thinned_params = []
    mse_connected_params = []
    vpt_thinned_params = []
    vpt_connected_params = []
    mse_thinned_score = None
    mse_connected_score = None
    vpt_thinned_score = None
    vpt_connected_score = None

    for combo in erdos_possible_combinations:
        # Check time and break if out of time
        t1 = time.time()
        if t1 - t0 > tf:
            # print("Break in Combo")
            break
        print(f"Here at {t1 - t0}")

        # Setup initial conditions
        n, p_thin, erdos_c, gamma, rho, sigma, alpha = combo

        mse_thinned = []
        mse_connected = []
        vpt_thinned = []
        vpt_connected = []

        for iter in range(iterations):
            print(f"Here - Iteration {iter}")
            print(f"Combo: {combo}")
            # Check time and break if out of time
            t1 = time.time()
            if t1 - t0 > tf:
                print("Break in iterations")
                break

            # Fixed random matrix
            W_in = np.random.rand(n, dim) - .5


            # Connected Matrix
            conn_prob = erdos_c / n + 1

            # Adjacency Matrix with Directed Erdos-Renyi adjacency matrix
            A_connected = nx.erdos_renyi_graph(n,conn_prob,directed=True)
            num_edges = len(A_connected.edges)
            A_connected = sparse.dok_matrix(nx.adjacency_matrix(A_connected).T)

            print(f"Post-connected-matrix")

            pred_states_connected, U_pred_connected = train_and_drive(A_connected, n, gamma, rho, sigma, W_in, u, U_train, t)

            # Compute VPT Score for comparison
            comp_array = np.linalg.norm(U_pred_connected.T - u(test_t))
            vpt_connected.append(np.argmax(comp_array > epsilon))

            print(f"Post-connected-pred")


            # Thinned matrix

            # Adjacency matrix with less edges
            A_thinned = remove_edges(A_connected, int((p_thin * num_edges)))

            pred_states_thinned, U_pred_thinned = train_and_drive(A_thinned, n, gamma, rho, sigma, W_in, u, U_train, t)

            # Compute VPT Score for comparison epsilon set to 5
            comp_array = np.linalg.norm(U_pred_thinned.T - u(test_t))
            vpt_thinned.append(np.argmax(comp_array > epsilon))

            print(f"Post-div calc")
            

            # Store diversity metrics

            # Store predictions and errors
            mse_thinned.append(np.mean((U_pred_thinned.T - u(test_t))**2))
            mse_connected.append(np.mean((U_pred_connected.T - u(test_t))**2))
    
        # print("Post break")

        # Check Means
        curr_mse_thinned_score = np.mean(mse_thinned)
        curr_mse_connected_score = np.mean(mse_connected)
        curr_vpt_thinned_score = np.mean(vpt_thinned)
        curr_vpt_connected_score = np.mean(vpt_connected)

        # Update params
        if mse_thinned_score is None or curr_mse_thinned_score < mse_thinned_score:
            mse_thinned_score = curr_mse_thinned_score
            mse_thinned_params = combo
        if mse_connected_score is None or curr_mse_connected_score < mse_connected_score:
            mse_connected_score = curr_mse_connected_score
            mse_connected_params = combo
        if vpt_thinned_score is None or curr_vpt_thinned_score > vpt_thinned_score:
            vpt_thinned_score = curr_vpt_thinned_score
            vpt_thinned_params = combo
        if vpt_connected_score is None or curr_vpt_connected_score > vpt_connected_score:
            vpt_connected_score = curr_vpt_connected_score
            vpt_connected_params = combo

    return mse_thinned_params, mse_connected_params, vpt_thinned_params, vpt_connected_params, mse_thinned_score, mse_connected_score, vpt_thinned_score, vpt_connected_score



"""
Main Method
"""

if __name__ == "__main__":
    # print("Start")
    erdos_possible_combinations = gridsearch_dict_setup()
    # print("Post Combo")
    t, U = get_patient_data(load_psg=False)
    results = run_gridsearch(erdos_possible_combinations, t, U)

    # Write the results 
    with open('reservoir_gridsearch_results/gridsearch_results_2.txt', 'w') as file:
        file.write("(n, p_thin, erdos_c, gamma, rho, sigma, alpha) score\n")
        file.write(f"mse_thinned: {results[0]}, {results[4]}\n")
        file.write(f"mse_connected: {results[1]}, {results[5]}\n")
        file.write(f"vpt_thinned: {results[2]}, {results[6]}\n")
        file.write(f"vpt_connected: {results[3]}, {results[7]}\n")