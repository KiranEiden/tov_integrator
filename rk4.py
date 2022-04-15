import warnings
import numpy as np

class System:
    """ Class defining a system of first-order ODEs. """
    
    def __init__(self, *rhsides, init_conds=None):
        """
        Takes a sequence of functions representing a system of differential equations, and
        optionally some initial conditions (which should include the independent variable as
        the first element).
        """
        
        self.rhsides = np.array(rhsides)
        self.ivar = None
        self.dvars = None
        
        if init_conds:
            self.set_conds(*init_conds)
        else:
            self.set_conds()
            
    def set_conds(self, *conds):
        """
        Sets the current state of the system. The first argument should be the value of the
        independent variable, and the remaining parameters should be the corresponding values of the
        dependent variables in the same order that their derivatives were supplied to the
        constructor.
        """
        
        if conds:
            self.ivar = conds[0]
            self.dvars = np.array(conds[1:], dtype=np.float64)
        else:
            self.ivar = 0.0
            self.dvars = np.zeros_like(rhsides, dtype=np.float64)
            

@np.vectorize
def call(func, *args):
    """ Calls all functions in an array with the given arguments, returning the results. """
    
    return func(*args)
    
def rk4(system, step, completed):
    """ Abstract implementation of classical 4th-order Runge-Kutta. """
    
    comp_level = 0
    
    while comp_level <= 0:
        
        with warnings.catch_warnings():
            
            # Eliminate the RuntimeWarnings from the nans
            warnings.simplefilter(action='ignore', category=RuntimeWarning)

            # Compute k-values
            k1 = step * call(system.rhsides, system.ivar, *system.dvars)
            k2 = step * call(system.rhsides, system.ivar + step / 2, *(system.dvars + k1 / 2))
            k3 = step * call(system.rhsides, system.ivar + step / 2, *(system.dvars + k2 / 2))
            k4 = step * call(system.rhsides, system.ivar + step, *(system.dvars + k3))
        
        if np.isnan([k1, k2, k3, k4]).any():
            # y went negative, meaning that the step was too large
            comp_level = -1
        else:
            # Try an update
            newvars = system.dvars + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            comp_level = completed(system.ivar + step, newvars)
        
        # Halve the step if necessary, otherwise update
        
        if comp_level < 0:
            step /= 2
        else:
            system.ivar += step
            system.dvars = newvars
            yield np.append([system.ivar], system.dvars)
