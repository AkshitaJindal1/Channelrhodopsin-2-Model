from pythonfiles.modeleq import opsinmodel_ODE
from pythonfiles.modelparams import setDefaultParameters_opsinmodel_ODE
from pyparsing import C
import data as data
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.optimize import fsolve 
from scipy import optimize
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from deap import base, creator, tools, algorithms
import random

#%%
# Load the dataset
# Load the dataset
def load_data():
    data_path = r'Chr2Model/data/Data_all.csv'
    # Load the CSV file
    data = pd.read_csv(data_path)
    # Rename the columns if necessary
    data.columns = ['t', 'I1', 'I2', 'I3', 'I4', 'I5']
    return data

def light_fun(t):
    return ((t >= 100) & (t <= 500)) * varpars[10]

# Load the experimental data
Data = load_data()

t = Data['t'].values
I_columns = Data[['I1', 'I2', 'I3', 'I4', 'I5']].values

# Set default parameters
modelpars, varpars, sysPars, fun = setDefaultParameters_opsinmodel_ODE()

# Define a function to compute predicted current based on ODE solution
def compute_I_opsin(t, varpars, light_fun):
    # Initial conditions
    y0 = [0.55, 0.225, 0.225]  # Initial conditions for O1, C2, O2
    y0 = fsolve(lambda y: fun(0, y, varpars, light_fun), y0)

    t_start = 0
    t_end = 600
    t_span = [t_start, t_end]
    dt = 10
    t_eval = np.arange(t_start, t_end, dt)

    sol = solve_ivp(fun, t_span, y0, t_eval=t_eval, args=(varpars, light_fun), rtol=1e-8, atol=1e-10)
    y = sol.y

    # Extract solutions
    O1 = sol.y[0]
    O2 = sol.y[2]
    
    # Compute predicted current (I_opsin)
    G = varpars[7]
    I_leak = varpars[8]
    gamma = 0.05
    E = 0
    V = -80
    
    I_opsin = G * (O1 + gamma * O2) * (V - E) - I_leak
    return I_opsin

# Define a function to compute residuals
def compute_residuals(time, observed_current, varpars, light_fun):
    I_opsin = compute_I_opsin(time, varpars, light_fun)
    residuals = (I_opsin - observed_current) ** 2  #Least squares
    return residuals

# Compute residuals for each current column
residuals_dict = {}
for idx, column in enumerate(['I1', 'I2', 'I3', 'I4', 'I5']):
    observed_current = Data[column].values
    residuals = compute_residuals(t, observed_current, varpars, light_fun)
    residuals_dict[column] = residuals.flatten()  # Flatten the residuals to ensure they are 1-dimensional

# Convert the residuals dictionary to a DataFrame
residuals_df = pd.DataFrame(residuals_dict)

# Save the residuals to a CSV file
residuals_df.to_csv('residuals_all.csv', index=False)

# Print the residuals (optional)
print(residuals_df)

# Plot the residuals
plt.figure(figsize=(12, 8))
for column in residuals_df.columns:
    plt.plot(t, residuals_df[column], label=column)

plt.xlabel('Time')
plt.ylabel('Residuals')
plt.title('Residuals for Each Current Dataset')
plt.legend()
plt.grid(True)
plt.savefig('residuals_plot.png')
plt.show()