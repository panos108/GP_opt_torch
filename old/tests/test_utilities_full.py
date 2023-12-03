import unittest
from SI_BOpt.src.utilities_full import BayesOpt
from SI_BOpt.optimization_toy_functions import Rosenbrock_cubic_line_constrained, Rosenbrock_disk_constrained
import numpy as np
import pyro
import sobol_seq

pyro.enable_validation(True)  # can help with debugging
pyro.set_rng_seed(0)
np.random.seed(0)

# Define optimization problem(s) for testing
opt_prob = Rosenbrock_cubic_line_constrained()
f1 = opt_prob.f_obj  # objective function
g1 = opt_prob.g1  # constraint 1
g2 = opt_prob.g2  # constraint 2
opt_sol = opt_prob.x_opt  # solution of the optimization problem
lower_b = opt_prob.lower_bound
upper_b = opt_prob.upper_bound

from scipy.optimize import minimize

fun = lambda x: f1(x)
cons = ({'type': 'ineq', 'fun': lambda x: -g1(x)},
        {'type': 'ineq', 'fun': lambda x: -g2(x)})
x0 = [0.6, 0.2]
bnds = ((lower_b[0], upper_b[0]), (lower_b[1], upper_b[1]))
res = minimize(fun, x0, method='SLSQP', tol=1e-6, bounds=bnds, constraints=cons)

# Parameters for BayesOpt
n_initial = 15  # int(3 * opt_prob.x_dim + 1)
X_initial = opt_prob.lower_bound + (opt_prob.upper_bound - opt_prob.lower_bound) * sobol_seq.i4_sobol_generate(
    opt_prob.x_dim, n_initial)  # np.random.rand(n_initial, opt_prob.x_dim)
Y_initial = np.array([
    [f1(X_initial[i]) for i in range(n_initial)],
    [g1(X_initial[i]) for i in range(n_initial)],
    [g2(X_initial[i]) for i in range(n_initial)]
]).T
n_func_eval = 15


class TestBayesOpt(unittest.TestCase):
    ''' Test 'BayesOpt' class in 'utilities_full.py' '''

    def setUp(self):
        self.check_tolerance = 4e-2

    # No known constraints cases
    def test_no_known_constr(self):
        n_func_eval = 15

        params_test = [
            {'objective': lambda x: (f1(x), [g1(x), g2(x)]), 'xo': lower_b, 'bounds': (lower_b, upper_b),
             'maxfun': n_func_eval, 'acquisition': 'LCB',
             'casadi': True, 'N_constraints': 2, 'known_constraints': None, 'X_initial': X_initial,
             'Y_initial': Y_initial, 'print_iteration': True, 'select_kernel': 'Matern52'},
            {'objective': lambda x: (f1(x), [g1(x), g2(x)]), 'xo': lower_b, 'bounds': (lower_b, upper_b),
             'maxfun': n_func_eval, 'acquisition': 'LCB',
             'casadi': True, 'N_constraints': 2, 'known_constraints': None, 'N_initial': n_initial,
             'print_iteration': True, 'select_kernel': 'Matern52'},

        ]

        for i in range(len(params_test)):
            result = BayesOpt().solve(**params_test[i])
            result_check = np.all(np.absolute(result.x - opt_sol) <= self.check_tolerance)
            print('not known cons: ', result.x)
            self.assertTrue(result_check)

    # known constraints case
    def test_known_constr(self):
        params_test = [
            # {'objective': lambda x: (f1(x), [g1(x)]), 'xo': lower_b, 'bounds': (lower_b, upper_b), 'maxfun': n_func_eval, 'acquisition': 'LCB',
            #  'casadi': True, 'N_constraints': 1, 'known_constraints': [g2], 'X_initial': X_initial,
            #  'Y_initial': Y_initial[:, :2], 'print_iteration':True},

            {'objective': f1, 'xo': lower_b, 'bounds': (lower_b, upper_b), 'maxfun': n_func_eval, 'acquisition': 'LCB',
             'casadi': True, 'N_constraints': 0, 'known_constraints': [g1, g2], 'X_initial': X_initial,
             'Y_initial': Y_initial[:, 0:1], 'print_iteration': True},

            {'objective': f1, 'xo': lower_b, 'bounds': (lower_b, upper_b), 'maxfun': n_func_eval, 'acquisition': 'LCB',
             'casadi': True, 'N_constraints': 0, 'known_constraints': [g1, g2], 'N_initial': n_initial,
             'print_iteration': True},

        ]

        for i in range(len(params_test)):
            result = BayesOpt().solve(**params_test[i])
            result_check = np.all(np.absolute(result.x - opt_sol) <= self.check_tolerance)
            self.assertTrue(result_check)

    # 'None' objective case
    def test_none_objective(self):
        params_test = [
            {'objective': None, 'xo': lower_b, 'bounds': (lower_b, upper_b), 'maxfun': n_func_eval,
             'acquisition': 'LCB',
             'casadi': True, 'N_constraints': 1, 'known_constraints': [g1], 'N_initial': n_initial},

            {'objective': None, 'xo': lower_b, 'bounds': (lower_b, upper_b), 'maxfun': n_func_eval,
             'acquisition': 'LCB', 'casadi': True, 'N_constraints': None, 'known_constraints': [g1, g2],
             'X_initial': X_initial, 'Y_initial': Y_initial[:, 0:1]},

            {'objective': None, 'xo': lower_b, 'bounds': (lower_b, upper_b), 'maxfun': n_func_eval,
             'acquisition': 'LCB', 'casadi': True, 'N_constraints': 2, 'known_constraints': None,
             'X_initial': X_initial, 'Y_initial': Y_initial},

            {'objective': None, 'xo': lower_b, 'bounds': (lower_b, upper_b), 'maxfun': n_func_eval,
             'acquisition': 'LCB', 'casadi': True, 'N_constraints': 1, 'known_constraints': [g2],
             'X_initial': X_initial, 'Y_initial': Y_initial[:, :2]},
        ]

        for i in range(len(params_test)):
            if 'N_initial' in list(params_test[i].keys()):
                self.assertRaises(ValueError)
            else:
                result = BayesOpt().solve(**params_test[i])
                self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()

