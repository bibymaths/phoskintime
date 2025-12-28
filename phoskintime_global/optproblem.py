import numpy as np
from pymoo.core.problem import ElementwiseProblem

from phoskintime_global.lossfn import jit_loss_core
from phoskintime_global.params import unpack_params
from phoskintime_global.simulate import simulate_odeint


class GlobalODE_MOO(ElementwiseProblem):
    """
    Elementwise multiobjective:
      F = [prot_mse, rna_mse, reg_loss]
    """

    def __init__(self, sys, slices, loss_data, defaults, lambdas, time_grid,
                 xl, xu, fail_value=1e12, elementwise_runner=None):
        super().__init__(
            n_var=len(xl),
            n_obj=3,
            n_ieq_constr=0,
            xl=xl,
            xu=xu,
            elementwise_runner=elementwise_runner
        )
        self.sys = sys
        self.slices = slices
        self.loss_data = loss_data
        self.defaults = defaults
        self.lambdas = lambdas
        self.time_grid = time_grid
        self.fail_value = float(fail_value)

    def _evaluate(self, x, out, *args, **kwargs):
        # 1) unpack + update
        p = unpack_params(x, self.slices)
        self.sys.update(**p)

        # 2) reg term (same as before, but keep as separate objective)
        reg = 0.0
        count = 0
        for k in ["A_i", "B_i", "C_i", "D_i", "E_i"]:
            diff = (p[k] - self.defaults[k]) / (self.defaults[k] + 1e-6)
            reg += float(np.sum(diff * diff))
            count += diff.size
        reg_loss = self.lambdas["prior"] * (reg / max(1, count))

        # 3) simulate (odeint + njit RHS + njit Jacobian)
        try:
            Y = simulate_odeint(
                self.sys,
                self.time_grid,
                rtol=1e-5,
                atol=1e-6,
                mxstep=5000
            )
        except Exception:
            out["F"] = np.array([self.fail_value, self.fail_value, self.fail_value], dtype=float)
            return

        if Y is None or Y.size == 0 or not np.all(np.isfinite(Y)):
            out["F"] = np.array([self.fail_value, self.fail_value, self.fail_value], dtype=float)
            return

        # Ensure (T, state_dim) contiguous
        Y = np.ascontiguousarray(Y)
        loss_p_sum, loss_r_sum = jit_loss_core(
            Y,
            self.loss_data["p_prot"], self.loss_data["t_prot"], self.loss_data["obs_prot"], self.loss_data["w_prot"],
            self.loss_data["p_rna"], self.loss_data["t_rna"], self.loss_data["obs_rna"], self.loss_data["w_rna"],
            self.loss_data["prot_map"], self.loss_data["rna_base_idx"]
        )

        prot_mse = loss_p_sum / self.loss_data["n_p"]
        rna_mse = loss_r_sum / self.loss_data["n_r"]

        out["F"] = np.array([prot_mse, rna_mse, reg_loss], dtype=float)
