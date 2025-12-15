import numpy as np
from numba import njit
import PoE.helper as helper

@njit(cache=True)
def pid_update(error, I_prev, prev_error, Kp, Ki, Kd, Ts, Kb, outputLimit_low, outputLimit_high, has_limit):
    """
    JIT-compiled PID update function.
    
    Returns: u_clipped, I_new, prev_error_new
    """
    # Update integral
    if Ki > 0:
        I_new = helper.discrete_Integrator(I_prev, error, Ts)
    else:
        I_new = I_prev.copy()
    
    # Derivative term
    d_err = helper.discrete_Derivative(prev_error, error, Ts)
    
    # PID output
    u = Kp * error + Ki * I_new + Kd * d_err
    
    # Apply output limits
    if has_limit:
        u_clipped = np.clip(u, outputLimit_low, outputLimit_high)
    else:
        u_clipped = u.copy()
    
    # Anti-windup correction
    if Ki > 0 and has_limit:
        aw_term = Kb * (u_clipped - u)
        I_new = I_new + aw_term
    
    return u_clipped, I_new, error.copy()


class PID_Discrete:
    def __init__(
        self,
        Kp,
        Ki,
        Kd,
        Ts,
        Kb=None,
        outputLimit=None,
        initial_integral=None,
        initial_prev_error=None
    ):
        """
        PID controller with anti-windup, back calculation, and initial state input.
        """
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.Ts = float(Ts)
        
        if outputLimit is not None:
            self.outputLimit_low = float(outputLimit[0])
            self.outputLimit_high = float(outputLimit[1])
            self.has_limit = True
        else:
            self.outputLimit_low = 0.0
            self.outputLimit_high = 0.0
            self.has_limit = False

        if Kb is None:
            if Ki > 0 and Kd > 0:
                self.Kb = 1.0 / np.sqrt(Ki * Kd)
            else:
                self.Kb = 1.0
        else:
            self.Kb = float(Kb)

        if initial_integral is not None:
            self._I = np.asarray(initial_integral, dtype=np.float64).ravel()
        else:
            self._I = None

        if initial_prev_error is not None:
            self._prev_error = np.asarray(initial_prev_error, dtype=np.float64).ravel()
        else:
            self._prev_error = None
        
        self._initialized = False

    def update(self, error):
        error = np.asarray(error, dtype=np.float64).ravel()

        # Initialize states on first call
        if self._I is None:
            self._I = np.zeros_like(error, dtype=np.float64)
        if self._prev_error is None:
            self._prev_error = error.copy()

        u_clipped, self._I, self._prev_error = pid_update(
            error, self._I, self._prev_error,
            self.Kp, self.Ki, self.Kd, self.Ts, self.Kb,
            self.outputLimit_low, self.outputLimit_high, self.has_limit
        )

        # Return scalar if input was scalar-like
        if u_clipped.size == 1:
            return u_clipped[0]
        return u_clipped
