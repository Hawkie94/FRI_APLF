import logging
import time
import pandas as pd
import numpy as np
import math

logger = logging.getLogger()
logger.setLevel(logging.INFO)

FILE_NAME = "./data/APLF_sample_data.csv"
HAS_HEADER = True

TS_COL = 3
CONS_COL = 2
TMP_COL = 0

TRAINING_DAYS = 180
DATA_DAILY_FREQUENCY = 24
TRAINING_LEN = TRAINING_DAYS * DATA_DAILY_FREQUENCY

L = 24
C = 48

s = 1
r = 3

lambda_s = 0.2
lambda_r = 0.7

W1 = 20
W2 = 80
W3 = 20

class Theta:
    def __init__(self, C, dim):
        self.eta = np.zeros((C, dim))
        self.sigma = np.zeros((C, 1))

class Gamma:
    def __init__(self, C, dim):
        self.gamma = np.zeros((C, dim))
        
        #self.prob = np.zeros((C, dim))
        self.prob = np.zeros((C, dim, dim))
        for i in range(C):
            self.prob[i] = np.eye(dim)

#N
#mean - mi | var - sigma
def N(x, mean, sd):
    prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density

#RMSE - Root mean square error
def RMSE(real, prediction):
    ret_val = np.sqrt(np.nanmean((real - prediction) ** 2))
    return ret_val

#MAPE - Mean average percentage error
def MAPE(real, estimated):
    ret_val = 100 * np.nanmean(np.abs(real - estimated) / real)
    return ret_val

def get_calendar_var_pos(inp):
    ts = pd.Timestamp(inp)
    day = ts.day_of_week
    hour = ts.hour

    if (day < 5):
        return hour
    else:
        return 24 + hour

def update_model(thetas, gammas, lambdas, x, y, t, t_vec, alpha1, alpha2):
    logger.info("Inside learn. Params are the following: \n\tx: %s \n\ty: %s \n\tt: %s", x, y, t)
    if alpha1 is None: alpha1 = 0
    if alpha2 is None: alpha2 = 0

    for i in range(L):
        c = get_calendar_var_pos(t_vec[i])

        us = np.ones((2, 1))
        us[1, 0] = y[i + 1][0]
        
        ur = np.ones((3, 1))
        ur[1, 0] = alpha1
        ur[2, 0] = alpha2

        u = {
            "s": us,
            "r": ur #x[i].T
        }

        for j in ['s', 'r']:
            eta, sigma, prob, gamma = update_params(
                eta = thetas[j].eta[c],
                sigma = thetas[j].sigma[c][0],
                P = gammas[j].prob[c],
                gamma = gammas[j].gamma[c][0],
                lamb = lambdas[j],
                s = y[i + 1],
                u = u[j]
            )

            thetas[j].sigma[c] = sigma
            gammas[j].prob[c] = prob
            gammas[j].gamma[c] = gamma
            
            index = 0
            for obj in eta:
                thetas[j].eta[c, index] = obj[0]
                index += 1
            #thetas[j].eta[0: , np.array((c))] = eta

def update_params(eta, sigma, P, gamma, lamb, s, u):
    gamma = 1 + (lamb * gamma)

    if np.size(P) > 1:
        if P.trace() > 10:
            #logger.warn("Value of P is bigger than 10. Check it out please!")
            P = np.eye(len(P))
        
        P = (1 / lamb) * (P - (P @ u @ u.T @ P) / (lamb + (u.T @ P @ u)))
        denominator = lamb + (u.T @ P @ u)
        sigma = np.sqrt(sigma ** 2 - (1 / gamma) * (sigma ** 2 - lamb * (s - (u.T @ eta)) ** 2) / denominator)
        eta = eta + ((P @ u).T[0] / denominator) * (s - u.T @ eta)
    else:
        if P > 10:
            #logger.warn("Value of P is bigger than 10. Check it out please!")
            P = np.array([[1]])

        P = (1 / lamb) * (P - (P * u * u.T * P) / (lamb + u.T * P * u))
        denominator = (lamb + u.T * P * u)
        sigma = np.sqrt(sigma ** 2 - (1 / gamma) * (sigma ** 2 - (lamb * (s - u * eta) ** 2) / denominator))
        eta = eta + ((P * u) / denominator) * (s - u * eta)
    
    return eta.T, sigma, P, gamma

def predict(thetas, s, r, t_vec, alpha1, alpha2):
    if alpha1 is None: alpha1 = 0
    if alpha2 is None: alpha2 = 0

    exp_st = s
    exp_e = 0
    v = np.array([0, 1])

    predictions = np.array([])
    errors = np.array([])

    for i in range(L):
        c = get_calendar_var_pos(t_vec[i])
        us = [1, exp_st]
        us = np.transpose(us)
        ur = np.array([1, alpha1, alpha2])

        eta_s = thetas['s'].eta[c]
        eta_r = thetas['r'].eta[c]
        sigma_s = thetas['s'].sigma[c]
        sigma_r = thetas['r'].sigma[c]
        
        tmp1 = sigma_r ** 2 + sigma_s ** 2 + ((v.T @ eta_s) ** 2) * exp_e
        new_st = (us.T @ eta_s * (sigma_r ** 2) + ur.T @ eta_r * (sigma_s ** 2 + ((v.T @ eta_s) ** 2) * exp_e ** 2)) / (sigma_r ** 2 + sigma_s ** 2 + ((v.T @ eta_s) ** 2) * exp_e)
        new_e = np.sqrt((sigma_r ** 2 * (sigma_s ** 2 + (v.T @ eta_s) * exp_e ** 2)) / (sigma_r ** 2 + sigma_s ** 2 + (v.T @ eta_s) ** 2 * exp_e ** 2))

        predictions = np.append(predictions, new_st, axis = 0)
        errors = np.append(errors, new_e, axis = 0)

        exp_st = new_st
        exp_e = new_e

    return predictions, errors

#TO DO - aplikovat W1 - W3 bez "zneuzitia" update_params - priemerne teploty by mali byt osobitne nastavene (!)
#TO DO - predvypocitat si alpha1, alpha2 pre cely dataset (!)
def main():
    #Theta and Gamma for loads and observations (s - loads, r - observations)
    theta_s = Theta(C, s + 1)
    theta_r = Theta(C, r)

    gamma_s = Gamma(C, s + 1)
    gamma_r = Gamma(C, r)

    thetas = {
        "s": theta_s,
        "r": theta_r
    }
    gammas = {
        "s": gamma_s,
        "r": gamma_r
    }

    #Phase 1 - TRAINING
    #Initializing variables to be 2D arrays (and removing the first value, so they are empty)
    loads = np.array([[0]])
    loads = loads[1:]
    observations = np.array([[0]])
    observations = observations[1:]
    dates = []

    f = open(FILE_NAME, "r")    
    if HAS_HEADER:
        f.readline()
    
    #Updating the prediction model with data from the selected range
    for i in range(TRAINING_LEN):
        line = f.readline().rstrip("\n")
        line_cols = line.split(',')
        
        cur_load = float(line_cols[CONS_COL])
        cur_observation = float(line_cols[TMP_COL])
        cur_date = line_cols[TS_COL] #get_calendar_var(line_cols[TS_COL])

        loads = np.append(loads, [[cur_load]], axis = 0)
        observations = np.append(observations, [[cur_observation]], axis = 0)
        dates.append(cur_date)

        #We read data and if we have at least (L + 1) values, we call the learning method ((L + 1) is required, because we are looking at time window of length L)
        if(len(loads) == (L + 1)):
            act_load = loads[0]
            load_vec = loads[0:(L + 1)]
            loads = loads[1:]
            
            act_observation = observations[0]
            observations = observations[1:]
            observation_vec = observations[0:L]
            
            act_date = dates[0]
            date_vec = dates[0:(L + 1)]
            dates = dates[1:]

            #x = [act_load, observation_vec]
            x = observation_vec
            y = load_vec

            update_model(
                thetas = thetas,
                gammas = gammas,
                lambdas = { "s": lambda_s, "r": lambda_r },
                x = x,
                y = y,
                t = act_date,
                t_vec = date_vec,
                alpha1 = None,
                alpha2 = None
            )
    
    #Phase 2 - PREDICTING
    predictions = []
    errors = []

    while True:
        line = f.readline()
        if not line:
            logging.info("INFO - end of file. Aborting reading.")
            break

        line_cols = line.rstrip("\n").split(',')
        
        cur_load = float(line_cols[CONS_COL])
        cur_observation = float(line_cols[TMP_COL])
        cur_date = line_cols[TS_COL]

        loads = np.append(loads, [[cur_load]], axis = 0)
        observations = np.append(observations, [[cur_observation]], axis = 0)
        dates.append(cur_date)

        if(len(loads) == (L + 1)):
            [tmp_preds, tmp_errs] = predict(
                thetas = thetas,
                s = loads[0][0],
                r = observations[1:],
                t_vec = dates[1:],
                alpha1 = None,
                alpha2 = None
            )

            predictions = np.append(predictions, tmp_preds, axis = 0)
            errors = np.append(errors, tmp_errs, axis = 0)

            loads = loads[1:]
            observations = observations[1:]
            dates = dates[1:]
    

    f.close()
    logger.info("INFO - ending script...")

if __name__ == "__main__":
    main()