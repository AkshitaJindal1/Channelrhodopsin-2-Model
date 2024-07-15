import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from deap import base, creator, tools, algorithms
import random
from pythonfiles.computeresidual import computeResidual
from pythonfiles.modeleq import opsinmodel_ODE
from pythonfiles.modelparams import setDefaultParameters_opsinmodel_ODE
from pythonfiles.computeresidual import light_fun
from scipy.optimize import fsolve 
from scipy import optimize
import seaborn as sns
import matplotlib.pyplot as plt
import os


# Function to load the data
def load_data():
    data = pd.read_csv('Data.csv')
    data.columns = ['t', 'I']
    return data

# Load the experimental data
Data = load_data()
t_exp = Data['t'].values
I_exp = Data['I'].values

#Extract the model
modelpars, varpars, sysPars, fun = setDefaultParameters_opsinmodel_ODE()

# Genetic Algorithm setup using DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Define realistic bounds for the parameters
param_bounds = [
    (0.01, 0.5),  # Gd1
    (0.01, 0.5),  # Gd2
    (1e-5, 1e-3), # Gr
    (0.01, 0.1),  # e12
    (0.01, 0.5),  # e21
    (0.1, 1.0),   # epsilon1
    (0.1, 1.0),   # epsilon2
    (0.1, 4),  # G
    (0.0, 0.5)   # I_leak
]

# Create random initial parameters within the defined bounds
def random_param(lower, upper):
    return random.uniform(lower, upper)

# Register the parameter creation function for each parameter
for i, (low, high) in enumerate(param_bounds):
    toolbox.register(f"attr_param{i}", random_param, low, high)

# Create individuals and population
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 [toolbox.attr_param0, toolbox.attr_param1, toolbox.attr_param2, toolbox.attr_param3,
                  toolbox.attr_param4, toolbox.attr_param5, toolbox.attr_param6, toolbox.attr_param7, toolbox.attr_param8], n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register genetic operators
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[low for low, _ in param_bounds], up=[up for _, up in param_bounds], eta= 1.0, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Update evaluate function to pass fun and Data
toolbox.register("evaluate", lambda ind: computeResidual(ind, fun, Data))

def main():
    population = toolbox.population(n=150)
    NGEN = 200
    CXPB, MUTPB = 0.9, 0.01   #Crossover probability and mutation probability

    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, CXPB, MUTPB)
        fits = toolbox.map(toolbox.evaluate, offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))
        top10 = tools.selBest(population, k=10)
        print(f"Generation {gen}: Best fitness {top10[0].fitness.values}")

    best_ind = tools.selBest(population, 1)[0]
    print(f"Best individual: {best_ind}")
    print(f"Best fitness: {best_ind.fitness.values}")
    best_params = best_ind

 # Save the best individual and its fitness value to a CSV file
    best_individual_df = pd.DataFrame([best_ind], columns=[f'param{i}' for i in range(len(best_ind))])
    best_individual_df['fitness'] = best_ind.fitness.values[0]
    best_individual_df.to_csv('best_individual.csv', index=False)
    

# Compute I_opsin using the best parameters
    varpars = [best_params[0], best_params[1], best_params[2], best_params[3], best_params[4], best_params[5], best_params[6], 0.0, 470, 6e-4, 0.77]
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

    G = best_params[7]
    I_leak = best_params[8]
    gamma = 0.05
    E = 0
    V = -80

    I_opsin = G * (O1 + gamma * O2) * (V - E) - I_leak
    I_interp = np.interp(Data['t'].values, t_eval, I_opsin)

    # Plot the experimental data and the fitted model
    plt.figure(figsize=(10, 6))
    plt.plot(t_exp, I_exp, 'o', label='Experimental Data')
    plt.plot(Data['t'].values, I_interp, '-', label='Fitted Model')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (I)')
    plt.legend()
    plt.title('Fitted Opsin Model vs Experimental Data')
    plt.savefig('optimisedplot.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()