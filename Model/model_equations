import numpy as np

def opsinmodel_ODE(t, y, vp, light):
    O1 = y[0]
    C2 = y[1]
    O2 = y[2]
    C1 = 1 - (O1 + O2 + C2)

    Gd1, Gd2, Gr, e12, e21, epsilon1, epsilon2, I, lambda_, sigma, w = vp
    I = light(t)

    Phi = sigma * I * lambda_ / w
    k1 = epsilon1 * Phi
    k2 = epsilon2 * Phi

    dO1_dt = (k1 * C1) + (e21 * O2) - O1 * (e12 + Gd1)
    dC2_dt = (Gd2 * O2) - C2 * (Gr + k2)
    dO2_dt = (k2 * C2) + (e12 * O1) - O2 * (e21 + Gd2)

    return [dO1_dt, dC2_dt, dO2_dt]
