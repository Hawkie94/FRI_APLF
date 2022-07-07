# -*- coding: utf-8 -*-
"""
Code for the paper Probabilistic Load Forecasting based on Adaptive Online Learning
https://github.com/MachineLearningBCAM/Load-forecasting-IEEE-TPWRS-2020/blob/master/APLF/APLF.py
@author: Verónica Álvarez
"""
import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import csv
import time
import pandas as pd

# import pandas as pd
#  data in .mat file
path = './data/'
#  os.chdir(path)
filename = '400buildings.mat'
mat = loadmat(path + filename)  # load mat-file
mdata = mat['data']  # variable in mat file
mdtype = mdata.dtype

data = {n: mdata[n][0, 0] for n in mdtype.names}
#energy_data = pd.read_csv('../Example/energy_temp_calendar.csv')

#  [MAPE, RMSE, predictions, load_demand, estimated_errors] = APLF(data, 300, 0.2, 0.7, 24, 48, 3)


class Theta:
    def __init__(self, c_par):
        C = c_par
        self.eta_s = np.zeros((2, C))
        self.sigma_s = np.zeros((1, C))
        self.eta_r = np.zeros((R, C))
        self.sigma_r = np.zeros((1, C))
        self.w_t = np.zeros((1, C))
        self.sigma_t = np.zeros((1, C))

class Gamma:
    def __init__(self, c_par, r_par):
        C, R = c_par, r_par
        self.gamma_t = np.zeros((1, C))
        self.P_t = np.zeros((1, C))
        self.gamma_s = np.zeros((1, C))
        self.P_s = np.zeros((C, 2, 2))
        self.gamma_r = np.zeros((1, C))
        self.P_r = np.zeros((C, R, R))
        for i in range(C):
            self.P_s[i] = np.eye(2)
            self.P_r[i] = np.eye(R)


def update_parameters(eta, sigma, P, gamma, lamb, s, mu):
    """
    Metehod from the bottom part of Algorithm 1
    :param eta:
    :param sigma:
    :param P:
    :param gamma:
    :param lamb:
    :param s:
    :param mu:
    :return:
    """
    # eq. (10)
    gamma = 1 + lamb * gamma
    if np.size(P) > 1:
        if P.trace() > 10:
            P = np.eye(len(P))

        # update the matrix P, eq. 9
        P = (1 / lamb) * (P - (np.dot(np.dot(np.dot(P, mu), mu.T), P) / (lamb + np.dot(np.dot(mu.T, P), mu))))
        # common denominator for computation efficiency
        denom_p = (lamb + np.dot(np.dot(mu.T, P), mu))
        # update sigma eq. 8
        sigma = np.sqrt(sigma ** 2 - (1 / gamma) * (sigma ** 2 - lamb * (s - np.dot(eta, mu)) ** 2) / denom_p)
        # update eta eq. 7
        eta = eta + np.dot(P, mu).T[0] / denom_p * (s - np.dot(mu.T, eta))
    else:
        # some Python thing ? ?
        if P > 10:
            P = 1
        # update the matrix P, eq. 9
        P = (1 / lamb) * (P - (P * mu * np.transpose(mu) * P) / (lamb + np.transpose(mu) * P * mu))
        denom_p = (lamb + mu * P * mu)
        # update sigma eq. 8 ,   (s - mu * eta) cab be separated as it is used in equation below
        sigma = np.sqrt(sigma ** 2 - (1 / gamma) * (sigma ** 2 - lamb * (s - mu * eta) ** 2) / denom_p)
        # update eta eq. 7
        eta = eta + (P * mu / denom_p ) * (s - mu * eta)
    # returns transposed eta, sigma, P and gamma
    return eta.T, sigma, P, gamma

# Algorithm 1
def update_model(theta, gamma, y, x, c, lambda_s, lambda_r):
    s0 = x[0]
    w = x[1:]
    y = [s0, y[0:]] # new loads vector
    L = len(y)  # size of y
    for i in range(L):
        # Update the mean of temperatures with the forgetting factor lambda = 1 and the feature vector u = 1
        theta.w_t[0][c[i]], theta.sigma_t[0, c[i]], gamma.P_t[0, c[i]], gamma.gamma_t[0, c[i]] = update_parameters(
            eta = theta.w_t[0, c[i][0]],
            sigma = theta.sigma_t[0, c[i][0]],
            P = gamma.P_t[0, c[i][0]],
            gamma = gamma.gamma_t[0, c[i][0]],
            lamb = 1,
            s = w[0][i][0],
            mu = 1
        )

        # Dummy variables
        alpha1, alpha2 = 0, 0
        if theta.w_t[0][c[i]] - w[0][i][0] > 20 and (w[0][i][0] > 80 or w[0][i][0] < 20):
            alpha1 = 1
        elif theta.w_t[0][c[i]] - w[0][i][0] < -20 and (w[0][i][0] > 80 or w[0][i][0] < 20):
            alpha2 = 1
        
        if alpha1 == 1 or alpha2 == 2:
            print("ok, something's wrong")
        # Feature vector that represents load
        us = np.ones((2, 1))
        # copy the first value
        us[1, 0] = y[1][i][0]
        # Update parameters denoted by s
        t, s, p, g = update_parameters(
            eta = theta.eta_s[0:, c[i][0]],
            sigma = theta.sigma_s[0, c[i][0]],
            P = gamma.P_s[c[i][0]],
            gamma = gamma.gamma_s[0, c[i][0]],
            lamb = lambda_s,
            s = y[1][i],
            mu = us
        )
        
        theta.eta_s[0:, c[i]] = t
        theta.sigma_s[0, c[i][0]] = s
        gamma.P_s[c[i][0]] = p
        gamma.gamma_s[0, c[i][0]] = g
        
        ur = np.ones((3, 1))
        ur[1, 0] = alpha1
        ur[2, 0] = alpha2
        # Update parameters denoted by r
        t, s, p, g = update_parameters(
            eta = theta.eta_r[0:, c[i][0]],
            sigma = theta.sigma_r[0][c[i][0]],
            P = gamma.P_r[c[i][0]],
            gamma = gamma.gamma_r[0][c[i][0]],
            lamb = lambda_r,
            s = y[1][i],
            mu = ur
        )

        theta.eta_r[0:, c[i]] = t
        theta.sigma_r[0][c[i]] = s
        gamma.P_r[c[i][0]] = p
        gamma.gamma_r[0][c[i]] = g
    return theta, gamma

def mape(real, estimated):
    return 100 * np.nanmean(np.abs(real - estimated) / real)
    #return np.nanmean(np.abs((real - estimated) / real)) * 100

def rmse(real, estimated):
    return np.sqrt(np.nanmean( (real - estimated)**2))

def prediction(theta, x, C):
    #  prediction function
    L = len(x[1])
    pred_s = np.zeros((L + 1, 1))
    e = np.zeros((L + 1, 1))
    pred_s[0, 0] = x[0]
    w = x[1:]
    for i in range(L):
        c = C[i]

        us = [1, pred_s[i, 0]]
        us = np.transpose(us)

        alpha1 = 0
        alpha2 = 0
        if theta.w_t[0][c] - w[0][i][0] > 20 and (w[0][i][0] > 80 or w[0][i][0] < 20):
            alpha1 = 1
        elif theta.w_t[0][c] - w[0][i][0] < -20 and (w[0][i][0] > 80 or w[0][i][0] < 20):
            alpha2 = 1
        ur = np.transpose([1, alpha1, alpha2])

        v = np.array([0, 1]).T

        eta_s = theta.eta_s[0:, c]
        eta_r = theta.eta_r[0:, c]
        sigma_s = theta.sigma_s[0][c]
        sigma_r = theta.sigma_r[0][c]

        pred_s[i + 1, 0] = \
          (np.dot(np.transpose(us), eta_s) * sigma_r ** 2 + np.dot(np.transpose(ur), eta_r) * (sigma_s ** 2 + np.dot(np.dot(v, eta_s) ** 2, e[i]))) / \
          (sigma_r * sigma_r + sigma_s ** 2 + np.dot((np.dot(v, eta_s) ** 2), e[i]))
        
        e[i + 1, 0] = np.sqrt( \
            (sigma_s ** 2 + np.dot(np.dot(np.dot(v, eta_s) ** 2, e[i]), sigma_r ** 2)) / \
            (sigma_r ** 2 + sigma_s ** 2 + np.dot(np.dot(v, eta_s) ** 2, e[i]))
        )
    
    return pred_s[1:], e[1:]

# MAPE =  6.925619458142164
# RMSE =  0.036035807611311275

# def APLF(data, days_train, lambdad, lambdar, L, C, R):
# [MAPE, RMSE, predictions, load_demand, estimated_errors] = APLF(data, 300, 0.2, 0.7, 24, 48, 3)
# days_train > 1 number of training days
days_train = 300
lambdad = 0.2  # forgetting factor
lambdar = 0.7  # forgetting factor
L = 24  # prediction horizon (hours)
C = 48  # length of the calendar information
R = 3  # length of feature vector of observations

#n = len(energy_data.energy_consumption)
#consumption = energy_data[['energy_consumption']].to_numpy()
#calendar_var = energy_data[['calendar']].to_numpy()
#calendar_var = calendar_var - 1
#temperature = energy_data[['temperature']].to_numpy()

n = len(data.get('consumption'))
consumption = data.get('consumption')
ct = data.get('c')
calendar_var = ct - 1
temperature = data.get('temperature')

n_train = 24 * days_train
theta = Theta(c_par=C)
gamma = Gamma(c_par=C, r_par=R)
predictions = []
estimated_errors = []
load_demand = []


for i in range(n_train - L + 1):
    s0 = consumption[i]
    w = temperature[i + 1:i + L + 1]
    x = [s0, w]
    y = consumption[i + 1:i + L + 1]
    cal = calendar_var[i + 1:i + L + 1]
    update_model(theta, gamma, y, x, cal, lambdad, lambdar)


for j in range(n_train + 1, n - L, L):
    s0 = consumption[j]
    w = temperature[j + 1:j + L + 1]
    x = [s0, w]
    [pred_s, e] = prediction(theta, x, calendar_var[j + 1:j + L + 1])
    predictions = np.append(predictions, np.transpose(pred_s))
    estimated_errors = np.append(estimated_errors, np.transpose(e))
    y = consumption[j + 1:j + L + 1]
    load_demand = np.append(load_demand, np.transpose(y))
    update_model(theta, gamma, y, x, calendar_var[j + 1:j + L + 1], lambdad, lambdar)


test_mape = mape(load_demand, predictions)
test_rmse = rmse(load_demand, predictions)
print(f'MAPE = {round(test_mape,2)} \nRMSE = {round(test_rmse,5)}')

# with open('results.csv', 'w+') as file:
#     writer = csv.writer(file)
#     writer.writerow(("predictions", "load demand", "estimated errors"))
#     rcount = 0
#     for i in range(len(predictions)):
#         writer.writerow((predictions[i], load_demand[i], estimated_errors[i]))
#         rcount = rcount + 1
#     file.close()

# return MAPE, RMSE, predictions, load_demand, estimated_errors
