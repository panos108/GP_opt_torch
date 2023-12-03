import numpy as np


class Rosenbrock_cubic_line_constrained():
    def __init__(self):
        self.x_dim = 2
        self.x_opt = np.array([1.0, 1.0])
        self.f_opt = 0.0
        self.n_constr = 2
        self.lower_bound = np.array([-1.5, -0.5])
        self.upper_bound = np.array([1.5, 2.5])
        self.constraints = [self.g1, self.g2]

    def f_obj(self, x):
        if x.shape[0] != self.x_dim:
            raise ValueError('expect 2 dimensional input')
        f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        return f

    def g1(self, x):
        ''' g1(x) <= 0'''
        if x.shape[0] != self.x_dim:
            raise ValueError('expect 2 dimensional input')
        g = (x[0] - 1) ** 3 - x[1] + 1
        return g

    def g2(self, x):
        '''g2(x) <= 0'''
        if x.shape[0] != self.x_dim:
            raise ValueError('expect 2 dimensional input')
        g = x[0] + x[1] - 2
        return g


class Rosenbrock_disk_constrained():
    def __init__(self):
        self.x_dim = 2
        self.x_opt = np.array([1.0, 1.0])
        self.f_opt = self.f_obj(self.x_opt)
        self.n_constr = 1
        self.lower_bound = np.array([-1.5, -1.5])
        self.upper_bound = np.array([1.5, 1.5])
        self.constraints = [self.g1]

    def f_obj(self, x):
        if x.shape[0] != self.x_dim:
            raise ValueError('expect 2 dimensional input')
        f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        return f

    def g1(self, x):
        ''' g1(x) <= 0'''
        if x.shape[0] != self.x_dim:
            raise ValueError('expect 2 dimensional input')
        g = x[0] ** 2 + x[1] ** 2 - 2
        return g


class Matyas():
    def __init__(self):
        self.x_dim = 2
        self.x_opt = np.array([0.0, 0.0])
        self.f_opt = 0.0
        self.n_constr = 0
        self.lower_bound = np.array([-10.0, -10.0])
        self.upper_bound = np.array([10.0, 10.0])
        self.constraints = []

    def f_obj(self, x):
        if x.shape[0] != self.x_dim:
            raise ValueError('expect 2 dimensional input')
        f = 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]
        return f


class Constrained_MIP():
    def __init__(self):
        self.x_dim = 2
        self.x_opt = np.array([1.7183, 1])
        self.f_opt = self.f_obj(self.x_opt)
        self.n_constr = 2
        self.lower_bound = np.array([0.57 + 1e-6, 0.])
        self.upper_bound = np.array([2., 1.])
        self.constraints = [self.g1, self.g2]

    def f_obj(self, x):
        if x.shape[0] != self.x_dim:
            raise ValueError('expect 2 dimensional input')
        f = -2.7 * x[1] + (x[0] ** 2)
        return f

    def g1(self, x):
        if x.shape[0] != self.x_dim:
            raise ValueError('expect 2 dimensional input')
        g = -np.log(1 + x[0]) + x[1]
        return g

    def g2(self, x):
        if x.shape[0] != self.x_dim:
            raise ValueError('expect 2 dimensional input')
        g = -np.log(x[0] - 0.57) - 1.1 + x[1]
        return g
