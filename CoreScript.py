import logging
import time
import pandas as pd
import numpy as np
import math

logger = logging.getLogger()
logger.setLevel(logging.INFO)

FILE_NAME = "APLF_sample_data.csv"
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
r = 1

lambda_s = 0.7
lambda_r = 0.3

class Theta:
    def __init__(self, C, dim):
        self.eta = np.zeros((C, dim))
        self.sigma = np.zeros((C, dim))

class Gamma:
    def __init__(self, C, dim):
        self.prob = np.zeros((C, dim))
        self.gamma = np.zeros((C, dim))

#N
#mean - mi | var - sigma
def N(x, mean, sd):
    prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density

def get_calendar_var_pos(inp):
    ts = pd.Timestamp(inp)
    day = ts.day_of_week
    hour = ts.hour

    if (day < 5):
        return hour
    else:
        return 24 + hour

def learn(theta, gamma, lambdas, x, y, t, t_vec):
    logger.info("Inside learn. Params are the following: \n\tx: %s \n\ty: %s \n\tt: %s", x, y, t)

    for i in range(L):
        c = get_calendar_var_pos(t_vec[i])
        u = {
            "s": np.array([1, y[i][0]]).T,
            "r": x[i].T #possible applying of PCA on the given vector
        }
        
        # @ => np.dot()
        # TO DO - pridat komentare (do buducna sa mozu zist)
        #       - oznacenie vzorcov podla cisiel Z CLANKU (!!!)
        for j in ['s', 'r']:
            #1
            cur_P = gamma[j].prob[c][0]
            gamma[j].prob[c][0] = (1 / lambdas[j]) * (
                cur_P - ((cur_P * (u[j] @ u[j].T) * cur_P) / (lambdas[j] + (u[j] @ u[j].T) * cur_P))
            )

            #2
            gamma[j].gamma[c][0] = 1 + lambdas[j] * gamma[j].gamma[c][0]

            #3
            #tmp variables (for faster references)
            cur_P = gamma[j].prob[c][0]
            cur_gamma = gamma[j].gamma[c][0]
            cur_sigma = theta[j].sigma[c][0]
            cur_eta = theta[j].eta[c][0]

            #small parts of the whole expression (for debugging)
            part1 = cur_eta @ u[j].T
            part2 = lambdas[j] * (y[i + 1][0] - u[j].T @ cur_eta) ** 2

            b = (lambdas[j] + np.dot(np.dot(u[j].T, cur_P), u[j]))
            tmp_val = (lambdas[j] * (y[i + 1][0] - np.dot(u[j].T, cur_eta)) ** 2) / (lambdas[j] + np.dot(np.dot(u[j].T, cur_P), u[j]))
            tmp_val_2 = np.sqrt(
                cur_sigma ** 2 - (1 / cur_gamma) * (
                    cur_sigma ** 2 - (
                        lambdas[j] * (y[i + 1][0] - np.dot(u[j].T, cur_eta)) ** 2) / (lambdas[j] + np.dot(u[j].T, u[j]) * cur_P
                    )
                )
            )
            
            #4
            theta[j].eta[c][0] = cur_eta + (cur_P * u[j]) / (lambdas[j] + u[j].T * cur_P * u[j]) * (y[i + 1] - np.dot(u[j].T * cur_eta))

def main():
    #Theta and Gamma for loads and observations (s - loads, r - observations)
    theta_s = Theta(C, s)
    theta_r = Theta(C, r)
    gamma_s = Gamma(C, s)
    gamma_r = Gamma(C, r)

    thetas = {
        "s": theta_s,
        "r": theta_r
    }
    gammas = {
        "s": gamma_s,
        "r": gamma_r
    }

    #Initializing variables to be 2D arrays (and removing the first value, so they are empty)
    loads = np.array([[0]])
    loads = loads[1:]

    observations = np.array([[0]])
    observations = observations[1:]

    dates = []

    f = open(FILE_NAME, "r")    
    if HAS_HEADER:
        f.readline()
    
    #Phase 1 - TRAINING
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

            learn(
                theta = thetas,
                gamma = gammas,
                lambdas = { "s": lambda_s, "r": lambda_r },
                x = x,
                y = y,
                t = act_date,
                t_vec = date_vec
            )


    f.close()
    logger.info("Ending script...")


if __name__ == "__main__":
    main()