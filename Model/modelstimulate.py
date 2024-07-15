from pythonfiles.modelparams import setDefaultParameters_opsinmodel_ODE
from scipy.integrate import odeint
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve 
import numpy as np

#%%

# Define light function
def light_fun(t):
    return (t >= 100) * (t <= 500) * varpars[7]

# Set default parameters
modelpars, varpars, sysPars, fun = setDefaultParameters_opsinmodel_ODE() 

# Define irradiance values
I_values = [0, 0.2, 0.4, 0.8, 1]  # Different Irradiance values

# Define colormap
cm = sns.color_palette("Blues", len(I_values) + 2)

# Define the initial guess
y0_initial = [0.55, 0.225, 0.225]  # Define initial conditions for O1, C2, and O2
y0 = fsolve(lambda y: fun(0, y, varpars, light_fun), y0_initial)

# Solve the ODE using solve_ivp solver
odeopts = {'rtol': 1e-8, 'atol': 1e-10}
sol = solve_ivp(fun, [0, 600], y0, method='RK45', t_eval=np.linspace(0, 600, 1000),
                args=(varpars, light_fun), **odeopts)
t = sol.t
y = sol.y

# Extract variables from solution
O1 = y[0]
O2 = y[2]

# Calculate opsin current
g0 = 4.0
G = 1.0551
gamma = 0.05
E = 0
V = -80  # holding potential

# Calculate I_opsin for the original Irradiance value
I_opsin1 = g0 * G * (O1 + gamma * O2) * (V - E)

# Plot1 original I_opsin
plt.figure()
plt.plot(t, I_opsin1, linewidth=2.0, color=cm[2])
plt.xlabel('Time')
plt.ylabel('I_{opsin}')
plt.ylim([-180, 10])
plt.title('Opsin Current')


for i in range(1, len(I_values)):
    # Update Irradiance value
    varpars[7] = I_values[i]

    # Redefine light function
    light_fun = lambda t: (t >= 100) * (t <= 500) * varpars[7]

    # Solve the ODE again for the current Irradiance value
    sol = solve_ivp(fun, [0, 600], y0, method='RK45', t_eval=np.linspace(0, 600, 1000),
                    args=(varpars, light_fun), **odeopts)
    O1 = sol.y[0]
    O2 = sol.y[2]

    # Calculate I_opsin for the current Irradiance value
    I_opsin = g0 * G * (O1 + gamma * O2) * (V - E)

    # Plot I_opsin with an offset in the y-axis for each Irradiance value
    plt.plot(t, I_opsin, linewidth=2.0, color=cm[i + 2])

plt.show()