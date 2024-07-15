from pythonfiles.modeleq import opsinmodel_ODE

def setDefaultParameters_opsinmodel_ODE():
    def fun(t, y, vp, light):
        return opsinmodel_ODE(t, y, vp, light)
        
    modelpars = {
        'Gd1': 0.1043,
        'Gd2': 0.3,
        'Gr': 5.494e-4,
        'e12': 0.079,
        'e21': 0.25,
        'epsilon1': 0.65,
        'epsilon2': 0.35,
        'I': 0,
        'lambda': 470,
        'sigma': 6e-4,
        'w': 0.77
    }

    varpars = [modelpars['Gd1'], modelpars['Gd2'], modelpars['Gr'], modelpars['e12'],
               modelpars['e21'], modelpars['epsilon1'], modelpars['epsilon2'], modelpars['I'],
               modelpars['lambda'], modelpars['sigma'], modelpars['w']]

    sysPars = {
        'tstart': 0,
        'tfinal': 100
    }
    return modelpars, varpars, sysPars, fun