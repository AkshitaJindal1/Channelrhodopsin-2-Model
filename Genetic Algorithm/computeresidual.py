from pythonfiles.modeleq import opsinmodel_ODE
from pythonfiles.modelparams import setDefaultParameters_opsinmodel_ODE
from pyparsing import C
import data as data
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve 
from scipy.signal import medfilt
from scipy import optimize
from scipy.optimize import least_squares
from os.path import join
from os.path import dirname
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from deap import base, creator, tools, algorithms
import random

#%%

## Load the experimental data
def load_data():
    # Define the path to the data file
    data = pd.read_csv('Data.csv')
    # Rename the columns
    data.columns = ['t', 'I']
    return data

# Load the experimental data
Data = load_data()

t = Data['t'].values
I = Data['I'].values

# Set default parameters
modelpars, varpars, sysPars, fun = setDefaultParameters_opsinmodel_ODE() 

def light_fun(t):
    return (t >= 100) * (t <= 500) * varpars[10]

def computeResidual(pars, fun, Data):
    varpars = [pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], 0.0, 0.0,  470, 6e-4, 0.77]
    y0 = [0.55, 0.225, 0.225]
    y0 = fsolve(lambda y: fun(0, y, varpars, light_fun), y0)

    t_start = 0
    t_end = 600
    t_span = [t_start, t_end]
    dt = 10
    t_eval = np.arange(t_start, t_end + dt, dt)

    sol = solve_ivp(fun, t_span, y0, t_eval=t_eval, args=(varpars, light_fun), rtol=1e-8, atol=1e-10)
    y = sol.y

    O1 = y[0, :]
    O2 = y[2, :]

    G = pars[6]
    I_leak = pars[7]
    gamma = 0.05
    E = 0
    V = -80

    I_opsin = G * (O1 + gamma * O2) * (V - E) - I_leak
    I_interp = np.interp(Data['t'].values, t_eval, I_opsin)

    res = np.sum((I_interp - Data['I'].values) ** 2)
    return res,