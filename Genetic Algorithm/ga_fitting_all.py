from pythonfiles.compute_residuals_all import computeResiduals
from pythonfiles.compute_residuals_all import light_fun
from pythonfiles.modeleq import opsinmodel_ODE
from pythonfiles.modelparams import setDefaultParameters_opsinmodel_ODE
from pyparsing import C
import data as data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from deap import base, creator, tools, algorithms
import random
import pickle  # Importing the pickle module

# Function to load the data
def load_data():
    data_path = r'Chr2Model/data/Data_all.csv'
    # Load the CSV file
    data = pd.read_csv(data_path)
    # Rename the columns if necessary
    data.columns = ['t', 'I1', 'I2', 'I3', 'I4', 'I5']
    return data

# Load the experimental data
Data = load_data()

t = Data['t'].values
I_columns = Data[['I1', 'I2', 'I3', 'I4', 'I5']].values

# Extract the model
modelpars, varpars, sysPars, fun = setDefaultParameters_opsinmodel_ODE()

def light_fun(t):
    return (t >= 100) * (t <= 500) * varpars[10]

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
    (0.1, 4),     # G
    (0.0, 0.5)    # I_leak
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
                  toolbox.attr_param4, toolbox.attr_param5, toolbox.attr_param6, toolbox.attr_param7], n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register genetic operators
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[low for low, _ in param_bounds], up=[up for _, up in param_bounds], eta=1.0, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Update evaluate function to pass fun and Data
toolbox.register("evaluate", lambda ind: computeResiduals(ind, fun, Data))

def main():
    # Create a dictionary to store optimized parameters and corresponding I_opsin for each column
    optimized_params = {}
    fitness_results = []
    fitness_generations = []

    # Loop over each column
    for col in range(I_columns.shape[1]):
        print(f"\nOptimizing parameters for column {col+1}")

        # Redefine evaluate function for each column
        def evaluate(individual):
            # Compute residuals for the individual and the column
            residuals = computeResiduals(individual, fun, Data[['t', f'I{col+1}']])
            # Minimize residuals (negative sign for minimization)
            return residuals[0],

        # Redefine the toolbox for each column
        toolbox.register("evaluate", evaluate)

        # Optimization
        population = toolbox.population(n=200)
        NGEN = 200
        CXPB, MUTPB = 0.9, 0.01

        fitness_gen = []  # To store fitness values for each generation

        for gen in range(NGEN):
            offspring = algorithms.varAnd(population, toolbox, CXPB, MUTPB)
            fits = toolbox.map(toolbox.evaluate, offspring)

            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

            population = toolbox.select(offspring, k=len(population))
            top10 = tools.selBest(population, k=10)
            print(f"Generation {gen}: Best fitness {top10[0].fitness.values}")
            
            # Store fitness values for this generation
            fitness_gen.append([gen] + [ind.fitness.values[0] for ind in population])

        fitness_generations.append(fitness_gen)

        best_ind = tools.selBest(population, 1)[0]
        print(f"Best individual: {best_ind}")
        print(f"Best fitness: {best_ind.fitness.values}")
        best_params = best_ind

        # Save optimized parameters and fitness to the dictionary and list
        optimized_params[col] = best_params
        fitness_results.append([col + 1] + best_params + [best_ind.fitness.values[0]])

    # Save the optimized parameters and fitness to a CSV file
    columns = ['Column'] + [f'Param{i}' for i in range(1, 9)] + ['Fitness']
    fitness_df = pd.DataFrame(fitness_results, columns=columns)
    fitness_df.to_csv('optimized_params_fitness_set1.csv', index=False)

    # Save the fitness for all generations to a CSV file
    fitness_gen_df = pd.DataFrame([item for sublist in fitness_generations for item in sublist], 
                                  columns=['Generation'] + [f'Fitness_col{i+1}' for i in range(I_columns.shape[1])])
    fitness_gen_df.to_csv('fitness_all_generations_set1.csv', index=False)

    # Plot the experimental data and the fitted model for each I column
    plt.figure(figsize=(12, 8))
    for col in range(I_columns.shape[1]):
        varpars = [optimized_params[col][0], optimized_params[col][1], optimized_params[col][2], optimized_params[col][3], 
                   optimized_params[col][4], optimized_params[col][5], 0.0, 0.0,  470, 6e-4, 0.77]
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

        G = optimized_params[col][6]
        I_leak = optimized_params[col][7]
        gamma = 0.05
        E = 0
        V = -80

        I_opsin = G * (O1 + gamma * O2) * (V - E) - I_leak
        I_interp = np.interp(Data['t'].values, t_eval, I_opsin)

        # Plot experimental data
        plt.plot(Data['t'].values, Data[f'I{col+1}'], 'bo', mfc='none', label=f'Experimental I{col+1}')

        # Plot fitted model
        plt.plot(Data['t'].values, I_interp, '-', label=f'Fitted Model I{col+1}')

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xlabel('Time (ms)')
    plt.ylabel('Photocurrent (pA)')
    plt.title('Fitted Opsin Model vs Experimental Data for All Columns')
    
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Move the legend to the left and remove the legend's box
    ax.legend(loc='lower right', frameon=False)

    plt.savefig('optimisedplot_all_columns_set1.png', dpi=300)
    plt.show()

    # Save the figure as a PNG file
    fig.savefig('optimisedplot_all_columns_all_set1.png', dpi=300)

    # Save the plot to a pickle file
    with open('figure_all_set1.pkl', 'wb') as f:
        pickle.dump(plt.gcf(), f)

    plt.show()

if __name__ == "__main__":
    main()