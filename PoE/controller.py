import numpy as np
import PoE.helper as helper

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
        Kp: proportional constant
        Ki: integral constant
        Kd: derivative constant
        Ts: sample time
        outputLimit: tuple (lower bound, upper bound)
        Kb: back calculation strength, default = 1/(sqrt(Ki*Kd))
        initial_state: initial error (or input) used for the integral and previous error states
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ts = Ts
        self.outputLimit = outputLimit

        if Kb is None:
            if Ki > 0 and Kd > 0:
                self.Kb = 1.0 / np.sqrt(Ki * Kd)
            else:
                self.Kb = 1.0
        else:
            self.Kb = Kb

        # integral state (I stores the time-summed error: I_k = sum_{j<=k} error_j * Ts)
        if initial_integral is not None:
            self._I = np.asarray(initial_integral, dtype=float).copy()
        else:
            self._I = None

        # previous error for derivative calculation
        if initial_prev_error is not None:
            self._prev_error = np.asarray(initial_prev_error, dtype=float).copy()
        else:
            self._prev_error = None

    def update(self, error):
        error = np.asarray(error, dtype=float)

        # initialize states on first call if they were not set by initial_state
        if self._I is None:
            self._I = np.zeros_like(error, dtype=float)
        if self._prev_error is None:
            self._prev_error = error.copy()

        if self.Ki > 0:
            self._I = helper.discrete_Integrator(self._I, error, self.Ts)

        # derivative term
        d_err = helper.discrete_Derivative(self._prev_error, error, self.Ts)

        # PID: compute (may be saturated below)
        u = self.Kp * error + self.Ki * self._I + self.Kd * d_err
        
        # Apply output limits (clamp, for anti-windup)
        if self.outputLimit is not None:
            u_clipped = np.clip(u, self.outputLimit[0], self.outputLimit[1])
        else:
            u_clipped = u

        # Anti-windup correction for integral (back calculation)
        if self.Ki > 0 and self.outputLimit is not None:
            aw_term = self.Kb * (u_clipped - u)
            self._I += aw_term

        # store for next iteration
        self._prev_error = error.copy()

        return u_clipped
