import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to
import sobol_seq
import pyDOE
import pyro
import pyro.contrib.gp as gp
import copy
from torch.distributions.multivariate_normal import MultivariateNormal as normal
import warnings
import numpy as np
from scipy.spatial.distance import cdist
from casadi import *
from SI_BOpt.src.extra_utils import MixedRBF, MixedMatern32, MixedMatern52, ConstrainedProblem
import pymoo.optimize
from pymoo.core.mixed import MixedVariableGA


np.random.seed(0)
torch.seed()


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'


class BayesOpt(object):
    def __init__(self):
        pyro.get_param_store().clear()
        self.obj_none_flag = False  # Added flag variable
        print('Start Bayesian Optimization using Torch')

    def solve(self, objective=None, xo=None, bounds=(0, 1), maxfun=20, N_initial=5,
              select_kernel='Matern32', acquisition='LCB', casadi=False, N_constraints=None,
              probabilistic=False, print_iteration=False, X_initial=None, Y_initial=None,
              known_constraints=None, idx_integer=None):

        """
        :param objective:           Objective function with numpy inputs
        :type objective:            Function
        :param xo:                  Initial point, not used at the moment
        :type xo:                   Numpy array
        :param bounds:              Bounds for the desicion variabls
        :type bounds:               Tuple of numpy arrays or numpy array
        :param maxfun:              Maximum number of function evaluations
        :type maxfun:               Integer
        :param N_initial:           Number of initial points
        :type N_initial:            Integer
        :param select_kernel:       Type of kernel
        :type select_kernel:        String
        :param acquisition:         Type of acquitiison functions
        :type acquisition:          String
        :param casadi:              Boolean that indicates if casadi is emplyed or not
        :type casadi:               Boolean
        :param constraints:         List of the constraints, if there are no constraints then Nana
        :type constraints:          List of functions
        :param probabilistic:       Satisfy constraints with probability
        :type probabilistic:        Boolean
        :param print_iteration:     Boolean that indicates if the results will be printed per iteration
        :type print_iteration:      Boolean
        :param X_initial            Initial inputs
        :type X_initial             Numpy array (N x input_dim)
        :param Y_initial            Initial objective & unknown constraint(s) evaluations
        :type Y_initial             Numpy array (N x (N_obj+N_constr))
        :param known_constraints    List of constraint(s) function with known closed-form expression
        :type known_constraints     List of function(s) with numpy inputs
        :param idx_integer          List of integer variables index
        :type idx_integer           List of integers
        :return:                    (x, objective, maxfun, dictionary with the results)
        :rtype:                     Tuple
        """

        if xo is None:
            self.x0 = torch.Tensor(bounds[0])
        else:
            self.x0 = torch.Tensor(xo)

        if known_constraints is None:
            known_constraints = []

        self.bounds = bounds
        self.maxfun = maxfun
        self.N_initial = N_initial
        self.objective = objective
        self.probabilistic = probabilistic
        self.print_iter = print_iteration
        self.known_constraints = known_constraints  # new attr. for known constraint(s)

        if objective is None:
            self.obj_none_flag = True  # assign flag=True if objective is 'None'

        if idx_integer is None:
            idx_integer = []
        else:
            print('GP with integer kernel is used, Use pymoo evo. algorithm for acquisition optimization')
            casadi = False

        self.idx_integer = idx_integer

        if N_constraints is None or N_constraints == 0:
            self.constraints = 0
            # constraints = 0  # this line is not used anywhere else
        else:
            self.constraints = N_constraints  # number of unknown constraint
            # We dont need this bit
            # constraints = len(self.constraints)

            if not (casadi) and acquisition != 'EIC' and (not idx_integer):
                casadi = True
                warnings.formatwarning = custom_formatwarning
                warnings.warn(
                    'WARNING: Pytorch optimization cannot handle constraints without EIC. Casadi and ipopt are used instead')

        self.set_functions = self.objective  # [self.objective]
        # self.set_functions += [*self.constraints]

        self.card_of_funcs = 1 + self.constraints  # len(self.set_functions), objective func + No. of unknown constraints

        self.kernel = select_kernel
        self.nx = max(self.x0.shape)
        self.casadi = casadi
        self.X_initial = X_initial
        self.Y_initial = Y_initial

        supported_acquisition = ['Mean', 'LCB', 'EI', 'EIC']
        k = 0
        for supp in supported_acquisition:
            if acquisition == supp:
                break
            else:
                k += 1
        if k == len(supported_acquisition):
            warnings.formatwarning = custom_formatwarning
            warnings.warn('WARNING: Selected acquisition does not exist, Lower Confidence Bound is used instead')
            acquisition = 'LCB'
        self.acquisition = acquisition

        if self.X_initial is None:
            #      self.X = torch.from_numpy(np.array([[ 0.14644051,  1.6455681 ],
            # [ 0.30829013,  1.13464955],
            # [-0.2290356 ,  1.43768234],
            # [-0.18723837,  2.175319  ],
            # [ 1.39098828,  0.65032456],
            # [ 0.87517511,  1.08668476],
            # [ 0.20413368,  2.27678991]]))#torch.from_numpy(np.random.rand(N_initial, self.nx) * (self.bounds[1] - self.bounds[0]) + self.bounds[0])#
            self.X = torch.from_numpy(
                sobol_seq.i4_sobol_generate(self.nx, N_initial) * (self.bounds[1] - self.bounds[0]) + self.bounds[
                    0])  # torch.from_numpy(pyDOE.lhs(self.nx, N_initial, criterion='maximin')) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

        else:
            self.X = torch.from_numpy(self.X_initial)
            self.Y = torch.from_numpy(self.Y_initial)

        if self.idx_integer:
            self.X[..., idx_integer] = torch.round(self.X[..., idx_integer])

        sol = self.run_main()
        return sol

    def run_initial(self):
        """
        This function computes the initial N_initial points to fit the Gaussian process
        :return: Y
        :rtype:  tensor
        """
        Y = torch.zeros([self.N_initial, self.card_of_funcs])
        for i in range(self.N_initial):
            # for j in range(self.card_of_funcs):
            Y[i, :] = self.compute_function(self.X[i, :], self.set_functions).reshape(-1, )
        return Y

    def define_GP(self, i):
        """
        This function predefines the Gaussian processes to be trained
        :param i:  This is the indexed GP to be trained 0 for objective and 1+ for the rest
        :type i:   integer
        :return:   Defined GP
        :rtype:    pyro object
        """
        pyro.get_param_store().clear()
        Y_unscaled = self.Y
        X_unscaled = self.X
        nx = self.nx

        # Scale the variables
        self.X_mean, self.X_std = X_unscaled.mean(axis=0), X_unscaled.std(axis=0)

        self.Y_mean, self.Y_std = Y_unscaled.mean(axis=0), Y_unscaled.std(axis=0)
        self.X_norm, self.Y_norm = (X_unscaled - self.X_mean) / self.X_std, \
                                   (Y_unscaled - self.Y_mean) / self.Y_std

        if self.idx_integer:
            self.X_norm[..., self.idx_integer] = X_unscaled[..., self.idx_integer]

        X, Y = self.X_norm, self.Y_norm

        if self.kernel == 'Matern52':
            if not self.idx_integer:
                gp_kernel = gp.kernels.Matern52(input_dim=nx, lengthscale=torch.ones(nx))
            else:
                gp_kernel = MixedMatern52(input_dim=nx, lengthscale=torch.ones(nx), integer_dims=self.idx_integer)

            gpmodel = gp.models.GPRegression(X, Y[:, i], gp_kernel,
                                             noise=torch.tensor(0.1), jitter=1.0e-4, )
        elif self.kernel == 'Matern32':
            if not self.idx_integer:
                gp_kernel = gp.kernels.Matern32(input_dim=nx, lengthscale=torch.ones(nx))
            else:
                gp_kernel = MixedMatern32(input_dim=nx, lengthscale=torch.ones(nx), integer_dims=self.idx_integer)

            gpmodel = gp.models.GPRegression(X, Y[:, i], gp_kernel,
                                             noise=torch.tensor(0.1), jitter=1.0e-4, )
        elif self.kernel == 'RBF':
            if not self.idx_integer:
                gp_kernel = gp.kernels.RBF(input_dim=nx, lengthscale=torch.ones(nx))
            else:
                gp_kernel = MixedRBF(input_dim=nx, lengthscale=torch.ones(nx), integer_dims=self.idx_integer)

            gpmodel = gp.models.GPRegression(X, Y[:, i], gp_kernel,
                                             noise=torch.tensor(0.1), jitter=1.0e-4, )
        else:
            print('NOT IMPLEMENTED KERNEL, USE RBF INSTEAD')
            if not self.idx_integer:
                gp_kernel = gp.kernels.RBF(input_dim=nx, lengthscale=torch.ones(nx))
            else:
                gp_kernel = MixedRBF(input_dim=nx, lengthscale=torch.ones(nx), integer_dims=self.idx_integer)

            gpmodel = gp.models.GPRegression(X, Y[:, i], gp_kernel,
                                             noise=torch.tensor(0.1), jitter=1.0e-4, )

        return gpmodel

    def training(self):
        """
        This function performs the training for the GPs
        :return: All GPs
        :rtype:  list of pyro objects
        """
        Y_unscaled = self.Y
        X_unscaled = self.X
        nx = self.nx

        # Scale the variables
        self.X_mean, self.X_std = X_unscaled.mean(axis=0), X_unscaled.std(axis=0)

        self.Y_mean, self.Y_std = Y_unscaled.mean(axis=0), Y_unscaled.std(axis=0)
        self.X_norm, self.Y_norm = (X_unscaled - self.X_mean) / self.X_std, \
                                   (Y_unscaled - self.Y_mean) / self.Y_std

        if self.idx_integer:
            self.X_norm[..., self.idx_integer] = X_unscaled[..., self.idx_integer]

        X, Y = self.X_norm, self.Y_norm
        # optimizers = []
        g = []
        for i in range(self.card_of_funcs):
            # self.gpmodel[i].set_data(X, Y[:,i])
            # # optimize the GP hyperparameters using Adam with lr=0.001
            # optimizers= torch.optim.Adam(self.gpmodel[i].parameters(), lr=0.001)
            # gp.util.train(self.gpmodel[i], optimizers)
            # self.gpmodel[i].requires_grad_(True)

            g.append((self.step_train(self.gpmodel[i], X, Y[:, i])))
            # self.gpmodel[i].requires_grad_(False)
        #     if i ==0:
        #         for j in (g[0].parameters()):
        #             print(j)
        # for k in (g[0].parameters()):
        #     print(k)
        # print('#------#')
        return g

    def step_train(self, gp_m, X, y):
        """
        This function performs the steps for the training of each gp
        :param gp_m:  The Gp to be trained
        :type gp_m:   pyro object
        :param X:     X input set to for training
        :type X:      tensor
        :param y:     Labels for the Gaussian processes
        :type y:      tensor
        :return:      trained GP
        :rtype:       pyro object
        """

        gp_m.set_data(X, y)
        # optimize the GP hyperparameters using Adam with lr=0.001
        pyro.get_param_store().clear()
        optimizers = torch.optim.Adam(gp_m.parameters(), lr=0.001)
        s = gp.util.train(gp_m, optimizers)

        return gp_m

    def acquisition_func(self, X_unscaled):
        """
        Given the input the acquisition function is computed FOR THE pytorch optimization
        :param X_unscaled:  input to be optimized
        :type X_unscaled:   SX
        :return:            acquisition as a casadi object
        :rtype:             SX
        """
        acquisition = self.acquisition
        X_unscaled = X_unscaled.reshape((1, -1))
        x = (X_unscaled - self.X_mean) / self.X_std
        if self.idx_integer:
            x[..., self.idx_integer] = X_unscaled[..., self.idx_integer]

        gp = (self.gpmodel[0])
        if acquisition == 'Mean':
            mu, _ = gp(x, full_cov=False, noiseless=False)
            ac = mu
        elif acquisition == 'LCB':
            mu, variance = gp(x, full_cov=False, noiseless=False)
            sigma = variance.sqrt()
            ac = mu - 2 * sigma
        elif acquisition == 'EI':
            mu, variance = gp(x, full_cov=False, noiseless=False)

            fs = self.f_min

            # Scale the variables
            Y_unscaled = self.Y[:, 0]
            Y_mean, Y_std = Y_unscaled.mean(), Y_unscaled.std()
            fs_norm = (fs - Y_mean) / Y_std
            mean = mu
            Delta = fs_norm - mean
            sigma = torch.sqrt(variance)

            Z = (Delta) / (sigma + 1e-5)
            norm_pdf = torch.exp(-Z ** 2 / 2) / torch.sqrt(2 * torch.Tensor([np.pi]))
            norm_cdf = 0.5 * (1 + torch.erf(Z / sqrt(2)))

            ac = -(sigma * norm_pdf + Delta * norm_cdf)
        elif acquisition == 'EIC':
            mu, variance = gp(x, full_cov=False, noiseless=False)

            fs = self.f_min

            # Scale the variables
            Y_unscaled = self.Y[:, 0]
            Y_mean, Y_std = Y_unscaled.mean(), Y_unscaled.std()
            fs_norm = (fs - Y_mean) / Y_std
            mean = mu
            Delta = fs_norm - mean
            sigma = torch.sqrt(variance)

            Z = (Delta) / (sigma + 1e-5)
            norm_pdf = torch.exp(-Z ** 2 / 2) / torch.sqrt(2 * torch.Tensor([np.pi]))
            norm_cdf = 0.5 * (1 + torch.erf(Z / sqrt(2)))

            ac = -(sigma * norm_pdf + Delta * norm_cdf)
            p = 1
            for i in range(1, self.card_of_funcs):
                Y_unscaled = self.Y[:, i]
                Y_mean, Y_std = Y_unscaled.mean(), Y_unscaled.std()
                gp_c = (self.gpmodel[i])

                mean_norm, var_norm = gp_c(x, full_cov=False, noiseless=False)

                mean, var = mean_norm * Y_std + Y_mean, var_norm * Y_std ** 2
                Z_p = mean / var  # mu/(variance + 1e-5)#(mu-Y_mean)/Y_std
                p *= 0.5 * (1 + torch.erf(Z_p / sqrt(2)))
            ac = ac * p

        else:
            print(NotImplementedError)
            ac = 0
        return ac

    def acquisition_func_ca(self, X_unscaled):
        """
        Given the input the acquisition function is computed FOR THE CASADI optimization
        :param X_unscaled:  input to be optimized
        :type X_unscaled:   SX
        :return:            acquisition as a casadi object
        :rtype:             SX
        """
        acquisition = self.acquisition
        # X_unscaled  = X_unscaled.reshape((1,-1))
        x = X_unscaled  # (X_unscaled - self.X_mean) / self.X_std
        gp = (self.gpmodel[0])
        if acquisition == 'Mean':
            mu, _ = self.GP_predict_ca(x, gp)
            ac = mu
        elif acquisition == 'LCB':
            mu, variance = self.GP_predict_ca(x, gp)
            sigma = sqrt(variance)
            ac = mu - 2 * sigma
        elif acquisition == 'EI':
            mu, variance = self.GP_predict_ca(x, gp)

            fs = self.f_min
            if fs == inf:
                ac = 0.
            else:
                Y_unscaled = self.Y[:, 0]
                Y_mean, Y_std = Y_unscaled.mean(), Y_unscaled.std()
                fs_norm = (fs - Y_mean) / (Y_std)
                fs_norm_ca = SX(fs_norm.data.numpy())
                mean = mu
                Delta = fs_norm_ca - mean
                sigma = sqrt(variance)

                Z = (Delta) / (sigma + 1e-5)
                norm_pdf = exp(-Z ** 2 / 2) / sqrt(2 * np.pi)
                norm_cdf = 0.5 * (1 + erf(Z / sqrt(2)))

                ac = -(sigma * norm_pdf + Delta * norm_cdf)
        elif acquisition == 'EIC':
            mu, variance = self.GP_predict_ca(x, gp)

            fs = self.f_min
            Y_unscaled = self.Y[:, 0]
            Y_mean, Y_std = Y_unscaled.mean(), Y_unscaled.std()
            fs_norm = (fs - Y_mean) / (Y_std)
            fs_norm_ca = SX(fs_norm.data.numpy())
            mean = mu
            Delta = fs_norm_ca - mean
            sigma = sqrt(variance)

            Z = (Delta) / (sigma + 1e-5)
            norm_pdf = exp(-Z ** 2 / 2) / sqrt(2 * np.pi)
            norm_cdf = 0.5 * (1 + erf(Z / sqrt(2)))

            ac = -(sigma * norm_pdf + Delta * norm_cdf)
            p = 1
            for i in range(1, self.card_of_funcs):
                Y_unscaled = self.Y[:, i]
                Y_mean, Y_std = SX(Y_unscaled.detach().numpy().mean()), \
                                SX(Y_unscaled.detach().numpy().std())
                gp_c = (self.gpmodel[i])
                mean_norm, var_norm = self.GP_predict_ca(x, gp_c)

                mean, var = mean_norm * Y_std + Y_mean, var_norm * Y_std ** 2
                Z_p = mean / var  # mu/(variance + 1e-5)#(mu-Y_mean)/Y_std
                p *= 0.5 * (1 + erf(Z_p / sqrt(2)))
            ac = ac * p
        else:
            print(NotImplementedError)
            ac = 0
        return ac

    def find_a_candidate(self, x_init):
        """
        Performs multistart optimization using BFGS within Pytorch
        :param x_init:  initial guess
        :type x_init:   tensor
        :return:        resulted optimum
        :rtype:         tensor detached from gradient
        """
        # transform x to an unconstrained domain
        constraint = constraints.interval(torch.from_numpy(self.bounds[0]).type(torch.FloatTensor),
                                          torch.from_numpy(self.bounds[1]).type(torch.FloatTensor))
        unconstrained_x_init = transform_to(constraint).inv(x_init)
        unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
        minimizer = optim.LBFGS([unconstrained_x], line_search_fn='strong_wolfe')

        def closure():
            minimizer.zero_grad()
            x = transform_to(constraint)(unconstrained_x)
            y = self.acquisition_func(x)
            autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
            return y

        minimizer.step(closure)
        # after finding a candidate in the unconstrained domain,
        # convert it back to original domain.
        x = transform_to(constraint)(unconstrained_x)
        return x.detach()

    def find_a_candidate_pymoo(self):

        constr_list = [lambda x_np, idx=i: self.gp_unscaled(x_np, idx, self.probabilistic)[0]
                       for i in range(1, self.card_of_funcs)]

        for g_known in self.known_constraints:
            constr_list.append(g_known)

        acq_f = lambda x_np: self.acquisition_func(torch.tensor(x_np).reshape(1, -1)).item()

        prob = ConstrainedProblem(x_dim=self.nx,
                                  x_bounds=self.bounds,
                                  integer_idx=self.idx_integer,
                                  f_objs=[acq_f],
                                  constraints=constr_list,
                                  )

        res = pymoo.optimize.minimize(problem=prob,
                                      algorithm=MixedVariableGA(pop=200),
                                      termination=('n_evals', 1500),
                                      seed=9,
                                      verbose=False
                                      )
        x_min_dict = {i: res.X[i] for i in range(len(res.X))}
        x_min = torch.tensor(list(x_min_dict.values())).reshape(1, -1)
        return x_min

    def gp_unscaled(self, x_np, idx, constr_prob=False):
        x_torch = torch.tensor(x_np, dtype=torch.double).reshape(1, -1)
        x_norm = (x_torch - self.X_mean) / self.X_std
        if self.idx_integer:
            x_norm[..., self.idx_integer] = x_torch[..., self.idx_integer]

        mean = (self.gpmodel[idx](x_norm)[0] * self.Y_std[idx] + self.Y_mean[idx]).item()
        var = (self.gpmodel[idx](x_norm)[1] * (self.Y_std[idx])**2).item()
        if constr_prob:
            mean += 1.5 * np.sqrt(var)
        return mean, var

    def find_a_candidate_ca(self, x_init):
        """
        Performs multistart optimization using ipopt within casadi
        :param x_init:  initial guess
        :type x_init:   tensor
        :return:        sol['x'].full(), solver, sol
        :rtype:         tSX solution, solver diagnosrtics, all the solution results
        """
        # transform x to an unconstrained domain
        constraint = constraints.interval(torch.from_numpy(self.bounds[0]).type(torch.FloatTensor),
                                          torch.from_numpy(self.bounds[1]).type(torch.FloatTensor))
        u_init = x_init.data.numpy()
        # x0 = self.model.x0#np.array([1., 150., 0.])
        # x = MX.sym('x_0',self.model.nd)

        nx = self.nx
        X = []
        gg = []
        w = []
        w0 = []
        lbw = []
        ubw = []
        xx = []
        yy = []
        zz = []
        g = []
        lbg = []
        ubg = []
        U = SX.sym('u_' + str(0), nx)
        w += [U]
        lbw.extend(self.bounds[0])
        ubw.extend(self.bounds[1])
        w0.extend(u_init)

        stdX, stdY, meanX, meanY = SX(self.X_std.detach().numpy()), \
                                   SX(self.Y_std.detach().numpy()), \
                                   SX(self.X_mean.detach().numpy()), \
                                   SX(self.Y_mean.detach().numpy())

        if self.card_of_funcs > 0 and self.acquisition != 'EIC':
            if self.known_constraints is None:
                self.known_constraints = []  # transform known_constraints attr. to empty list for checking if constr. is known in iteration

            for i in range(1, self.card_of_funcs):  # self.model.nk*self.model.nu)

                gp_c = (self.gpmodel[i])

                mean_norm, var_norm = self.GP_predict_ca(U, gp_c)

                mean, var = mean_norm * stdY[i] + meanY[i], var_norm * stdY[i] ** 2

                if self.probabilistic:
                    g += [mean + 2 * (var) ** 0.5]  # + slack]
                else:
                    g += [mean]  # + slack]

                xx += [mean]
                yy += [mean + 2 * (var) ** 0.5]

                lbg.extend([-inf])
                ubg.extend([0.])

            for i in range(len(self.known_constraints)):
                g += [self.known_constraints[i](U)]
                lbg.extend([-inf])
                ubg.extend([0.])

                # loop over the constraints

                # lbg.extend([-inf])
                # ubg.extend([0.])
        # delay = 0
        # for k in range(2,152):
        #     #expon = exp((U[2]+1.001)*log((k-1)/150))
        #     blue   = U[0]+(U[1]-U[0])*mpower(((k-1)/150),(U[2]+1.001))#U[2]**log(((k-1)/150))#
        #     delay += 6.3597*blue*60/24-6618
        #     delay += 4.5497*(blue*60/24) - 6278.9
        # g += [delay/2000000-1]
        # lbg.extend([-inf])
        # ubg.extend([0.])

        # g += [sum1((U-uk.reshape(-1,1))**2)-self.Delta0 ** 2]
        # lbg.extend([-inf])
        # ubg.extend([0.])
        obj = self.acquisition_func_ca(U)
        # if self.obj == 1:
        #     obj = mean_obj_model(U) + mean_obj(U)
        # elif self.obj == 2:
        #     obj = mean_obj_model(U) + mean_obj(U) \
        #           - 3 * sqrt(var_obj(U))  # + 1e4*Sum_slack
        # elif self.obj == 3:
        #     fs          = self.obj_min
        #     obj_f       = mean_obj(U)
        #     obj_f_model = mean_obj_model(U)

        # mean =  obj_f_model+ obj_f
        # Delta = fs - mean - 0.01
        # sigma = sqrt(var_obj(U)+var_obj_model(U))
        # Delta_p = np.max(mean(X) - fs)
        # if sigma == 0.:
        #     Z = 0.
        # else:
        # Z = (Delta) / sigma
        # norm_pdf = exp(-Z**2/2)/sqrt(2*np.pi)
        # norm_cdf = 0.5 * (1+ erf(Z/sqrt(2)))
        # obj = -(sigma * norm_pdf + Delta * norm_cdf)
        # yy += [mean_obj_model(U)]
        # zz += [mean_obj(U)]
        opts = {}
        opts["expand"] = True
        opts["ipopt.print_level"] = 0
        opts["ipopt.max_iter"] = 1000
        opts["ipopt.tol"] = 1e-8
        opts["calc_lam_p"] = False
        opts["calc_multipliers"] = False
        opts["ipopt.print_timing_statistics"] = "no"
        opts["print_time"] = False
        problem = {'f': obj, 'x': vertcat(*w), 'g': vertcat(*g)}

        # trajectories = Function('trajectories', [vertcat(*w)]
        #                        , [U, horzcat(*X), horzcat(*gg), horzcat(*xx), horzcat(*yy), horzcat(*zz)],
        #                        ['w'], ['x', 'u', 'gg', 'gpc', 'objm', 'gpobj'])     #comment this part bcs. it's not used when constr. is known

        solver = nlpsol('solver', 'ipopt', problem,
                        opts)  # , "ipopt.max_iter": 1e4})  # , {"ipopt.hessian_approximation":"limited-memory"})#, {"ipopt.tol": 1e-10, "ipopt.print_level": 0})#, {"ipopt.hessian_approximation":"limited-memory"})
        # Function to get x and u trajectories from w

        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        return sol['x'].full(), solver, sol

    def next_x(self, num_candidates=40):
        """
        Performs the multistart for optimization
        :param num_candidates:  number of candidates
        :type num_candidates:   integer
        :return:                best solution
        :rtype:                 tensor
        """
        candidates = []
        values = []

        # x_init = self.gpmodel.X[-1:]
        x_init = torch.from_numpy(self.generate_samples_for_multistart(num_candidates)).type(torch.FloatTensor)
        for i in range(num_candidates):
            # Find the minimum
            self.f_min = self.find_min_so_far()
            if self.casadi:

                x, solver, sol = self.find_a_candidate_ca(x_init[i])
                x = torch.from_numpy(x)
                if solver.stats()['return_status'] == 'Solve_Succeeded' or solver.stats()[
                    'return_status'] == 'Solved To Acceptable Level':
                    y = torch.Tensor(DM(self.acquisition_func_ca(x.data.numpy())).full())
                else:
                    y = torch.Tensor([[inf]])  # torch.from_numpy(np.array([np.inf]))
            else:
                if not self.idx_integer:
                    x = self.find_a_candidate(x_init[i])
                else:
                    x = self.find_a_candidate_pymoo()

                y = self.acquisition_func(x)

            candidates.append(x)
            values.append(y)

            if self.idx_integer:
                break  # x is already selected from candidates within evol. opt already

        # x_plot = torch.Tensor(np.linspace(0, 1, 100))
        # y_plot = torch.Tensor(np.linspace(0, 1, 100))
        #
        # for i in range(100):
        #     y_plot[i] = self.acquisition_func(x_plot[[i]])
        # plt.plot(x_plot,y_plot.detach().numpy())
        # print('--')
        # print(self.gpmodel.kernel.lengthscale_unconstrained)
        # print(self.gpmodel.kernel.variance_unconstrained)
        argmin = torch.min(torch.cat(values), dim=0)[1].item()
        return candidates[argmin]

    def generate_samples_for_multistart(self, multi_start=30):
        """
        Generates the points for the multistart
        :param multi_start:  number of multistarts
        :type multi_start:   integer
        :return:             initial guesses for optimization
        :rtype:              numpy array
        """
        multi_startvec = sobol_seq.i4_sobol_generate(self.nx, multi_start)

        multi_startvec_scaled = multi_startvec * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

        return multi_startvec_scaled

    def update_data(self, xmin):
        """
        Updates the data set using the new optimum point
        """
        xmin = xmin.reshape(-1, )
        y = torch.zeros([1, self.card_of_funcs])
        # for i in range(self.card_of_funcs):
        y[0, :] = self.compute_function(xmin, self.set_functions).reshape(-1, )
        if not (np.linalg.norm(self.X - xmin.reshape(1, self.nx), axis=1) < 1e-5).any():
            self.X = torch.cat([self.X, xmin.reshape(1, self.nx)])
            self.Y = torch.cat([self.Y, y])
        else:
            print('A previous point was found to be optimal')

    def run_main(self):
        """
        Run the main loop of the Bayesopt
        :return: x_opt, y_opt, self.maxfun, output_dict
        :rtype:  tensor, tensor, integer, dictionary
        """
        if self.X_initial is None:
            if self.obj_none_flag == True:  # to handle when obj. func.is 'None' and initial points are not provided.
                raise TypeError('Objective function is \'None\', and cannot be evaluated. '
                                'Try using initial points (X_initial,Y_initial) instead of N_initial.')
            else:
                self.Y = self.run_initial()

        else:
            print('Initial points were given')

        self.gpmodel = []
        for i in range(self.card_of_funcs):
            self.gpmodel += [(self.define_GP(i))]

        self.gpmodel = self.training()
        x_his_optimal = np.zeros([self.maxfun, self.nx])
        y_his_optimal = np.zeros([self.maxfun, self.card_of_funcs])

        for i in range(self.maxfun):
            xmin = self.next_x()  # Here we find the new point
            if self.idx_integer:
                xmin[..., self.idx_integer] = torch.round(xmin[..., self.idx_integer])

            if self.obj_none_flag:
                print('Objective function is \'None\'')
                print('Next x_min to evaluate: ', xmin.numpy())
                break  # Break if objective function is 'None'

            # print(xmin)
            self.update_data(xmin)  # Here we go and evaluate the *real* objective and constraints

            if self.print_iter:
                _, optim_iter = self.find_min_so_far(argmin=True)  #
                x_print = self.X[[optim_iter]]
                y_print = self.Y[[optim_iter], 0]

                print(f'nex x: {xmin.numpy()} at iter {i+1}')
                print('best x: ', x_print.data.numpy(), 'at iter ', i + 1)
                print('best obj: ', y_print.data.numpy(), 'at iter ', i + 1)
                x_his_optimal[i, :] = self.X[optim_iter].data.numpy()
                y_his_optimal[i, :] = self.Y[optim_iter, :].data.numpy()

            self.gpmodel = self.training()

        _, optim = self.find_min_so_far(argmin=True)
        x_opt = self.X[[optim]]
        y_opt = self.Y[[optim], 0]

        print('Optimum Objective Found: ', y_opt.numpy())
        print('Optimum point Found: ', x_opt.numpy())
        output_dict = {}
        # output_dict['x_all'] = self.X
        # output_dict['x']     = x_opt#
        # output_dict['f']     = y_opt#
        # if self.card_of_funcs>0:
        #     output_dict['g'] = self.Y[optim,1:]
        # else:
        #     output_dict['g'] = 'No constraints'
        # output_dict['g_store'] = self.Y[self.N_initial:,1:]
        # output_dict['x_store'] = self.X[self.N_initial:]
        # output_dict['f_store'] = self.Y[self.N_initial:,0]
        # output_dict['f_all']   = self.Y
        # output_dict['f_best_so_far'] = x_his_optimal
        # output_dict['x_best_so_far'] = y_his_optimal

        # f_so_far, g_so_far = self.find_min_so_far()
        output_dict = {}
        output_dict['g_store'] = self.Y[:, 1:]
        output_dict['x_store'] = self.X
        output_dict['f_store'] = self.Y[:, 0:1]
        output_dict['N_evals'] = self.maxfun
        output_dict['g_best_so_far'] = y_his_optimal[:, 1:]
        output_dict['f_best_so_far'] = y_his_optimal[:, 0]
        output_dict['x_best_so_far'] = x_his_optimal
        output_dict['samples_at_iteration'] = np.arange(1, 1 + len(output_dict['f_store']))
        output_dict['TR'] = [None] * self.maxfun
        output_dict['gp_model'] = [lambda x_np, idx=i: self.gp_unscaled(x_np, idx) for i in range(self.card_of_funcs)]

        if self.obj_none_flag:
            output_dict['next_x'] = xmin.numpy()

        return solutions(x_opt, y_opt, self.maxfun, output_dict)

    def find_min_so_far(self, argmin=False):
        """
        This function find the best solution so far, mainly used for EI
        :param argmin: Boolean that if it is True the func returns which point is the best
        :type argmin:  Boolean
        """
        min = np.inf
        index = np.inf
        if self.known_constraints is None:
            if self.card_of_funcs == 1:
                for i in range(len(self.X)):
                    y = self.Y[i, 0]
                    if y < min:
                        min = y
                        index = i
            else:
                for i in range(len(self.X)):
                    y = self.Y[i, 0]
                    if y < min and all(self.Y[i, 1:].data.numpy() <= 1e-3):
                        min = y
                        index = i
        else:
            if self.card_of_funcs == 1:
                for i in range(len(self.X)):
                    y = self.Y[i, 0]
                    X_ca = DM(
                        self.X[i, :].data.numpy())  # HERE I transform the tensor to casadi symbolic to be evaluated
                    # next the casadi is going to numpy array
                    eval_known = [np.array(self.known_constraints[j](X_ca)).squeeze() <= 1e-3
                                  for j in range(len(self.known_constraints))]
                    if y < min and all(eval_known):
                        min = y
                        index = i
            else:
                for i in range(len(self.X)):
                    y = self.Y[i, 0]
                    X_ca = DM(
                        self.X[i, :].data.numpy())  # HERE I transform the tensor to casadi symbolic to be evaluated
                    # next the casadi is going to numpy array
                    eval_known = [np.array(self.known_constraints[j](X_ca)).squeeze() <= 1e-3
                                  for j in range(len(self.known_constraints))]
                    if y < min \
                            and all(self.Y[i, 1:].data.numpy() <= 1e-3) \
                            and all(eval_known):
                        min = y
                        index = i
        if min == inf:
            for i in range(len(self.X)):
                y = self.Y[i, 0]
                if y < min:
                    min = y
                    index = i
        if argmin:
            return min, index
        else:
            return min

    def compute_function(self, x_torch, f):
        """
        Computes the function evaluation. It goes from tensor to numpy, then computes the function and
        comes transforms the solution from numpy to tensors
        :param x_torch:  input
        :type x_torch:   tensor
        :param f:        object to be evaluated
        :type f:         function
        :return:         evaluated point
        :rtype:          tensor
        """
        x = x_torch.detach().numpy().reshape(-1, )
        if self.card_of_funcs > 1:
            obj_temp, con_temp = f(x)

            y = np.array([obj_temp, *con_temp]).reshape(-1, )
        else:
            obj_temp = f(x)

            y = np.array([obj_temp]).reshape(-1, )

        return torch.from_numpy(y).type(torch.FloatTensor)

    def extract_parameters(self, gp):
        """
        Extracts the parameters of the gp
        :param gp:  the gp that we need to extract the parameters from
        :type gp:   pyro object
        :return:    list of parameters
        :rtype:     list
        """
        params = []

        for param in gp.kernel.parameters():
            params += [np.exp(param.data.numpy()).astype(np.double)]
        params += [np.exp(gp.noise_unconstrained.data.numpy().astype(np.double))]
        return params

    def calc_cov_sample(self, xnorm, Xnorm, ell, sf2):
        '''
        Calculates the covariance of a single sample xnorm against the dataset Xnorm
        --- decription ---
        '''
        # internal parameters
        nx_dim = self.nx
        kernel = self.kernel
        if kernel == 'RBF':
            dist = cdist(Xnorm, xnorm.reshape(1, nx_dim), 'seuclidean', V=ell) ** 2
            cov_matrix = sf2 * np.exp(-.5 * dist)
        elif kernel == 'Matern32':
            dist = cdist(Xnorm, xnorm.reshape(1, nx_dim), 'seuclidean', V=ell)
            t_1 = 1 + 3 ** 0.5 * dist
            t_2 = np.exp(-3 ** 0.5 * dist)
            cov_matrix = sf2 * t_1 * t_2
        elif kernel == 'Matern52':
            dist = cdist(Xnorm, xnorm.reshape((1, nx_dim)), 'seuclidean', V=ell)
            t_1 = 1 + 5 ** 0.5 * dist + 5 / 3 * dist ** 2
            t_2 = np.exp(-5 ** 0.5 * dist)
            cov_matrix = sf2 * t_1 * t_2
        else:
            print('ERROR no kernel with name ', kernel)
            cov_matrix = 0.

        return cov_matrix

    def Cov_mat(self, kernel, X_norm, W, sf2):
        '''
        Calculates the covariance matrix of a dataset Xnorm
        --- decription ---
        '''
        dist = cdist(X_norm, X_norm, 'seuclidean', V=W) ** 2
        r = np.sqrt(dist)
        if kernel == 'RBF':
            cov_matrix = sf2 * np.exp(-0.5 * dist)
            return cov_matrix
            # Note: cdist =>  sqrt(sum(u_i-v_i)^2/V[x_i])
        elif kernel == 'Matern32':
            cov_matrix = sf2 * (1 + 3 ** 0.5 * r) * np.exp(-r * 3 ** 0.5)
            return cov_matrix
        elif kernel == 'Matern52':
            cov_matrix = sf2 * (1 + 5 ** 0.5 * r + 5 / 3 * r ** 2) * np.exp(-r * 5 ** 0.5)
            return cov_matrix
        else:
            print('ERROR no kernel with name ', kernel)

    def GP_predict_ca(self, x, gp):  # , X,Y):
        gp1 = (gp)
        params = self.extract_parameters(gp1)
        X_norm = self.X_norm.detach().numpy()
        Y_norm = gp1.y.detach().numpy()  # self.Y_norm.detach().numpy()
        nd, hypopt = 1, params
        Kopt = self.Cov_mat(self.kernel, X_norm, hypopt[1] ** 2, hypopt[0]) \
               + params[-1] * np.eye(X_norm.shape[0]) + 1e-5 * np.eye(
            X_norm.shape[0])  # gp1.kernel(self.X_norm).detach().numpy()\

        invKopt = np.linalg.pinv(Kopt)
        Ynorm, Xnorm = SX(DM(Y_norm)), SX(DM(X_norm))
        ndat = Xnorm.shape[0]
        nX, covSEfcn = self.nx, self.covSEard(kernel=self.kernel)
        stdX, stdY, meanX, meanY = SX(self.X_std.detach().numpy()), \
                                   SX(self.Y_std.detach().numpy()), \
                                   SX(self.X_mean.detach().numpy()), \
                                   SX(self.Y_mean.detach().numpy())
        #        nk     = 12
        # x = SX.sym('x', nX)
        # nk     = X.shape[0]
        xnorm = (x - meanX) / stdX
        # Xnorm2 = (X - meanX)/stdX
        # Ynorm2 = (Y - meanY)/stdY

        k = SX.zeros(ndat)
        # k2     = SX.zeros(ndat+nk)
        mean = SX.zeros(nd)
        mean2 = SX.zeros(nd)

        var = SX.zeros(nd)
        # Xnorm2 = SX.sym('Xnorm2',ndat+nk,nX)
        # invKY2 = SX.sym('invKY2',ndat+nk,nd)

        invK = SX(DM(invKopt))
        # hyper = SX(DM(hypopt))
        ellopt, sf2opt = hypopt[1] ** 2, hypopt[0]
        for j in range(ndat):
            k[j] = covSEfcn(xnorm, Xnorm[j, :], ellopt, sf2opt)
        # for j in range(ndat+nk):
        #    k2[j] = covSEfcn(xnorm,Xnorm2[j,:],ellopt,sf2opt)
        # self.calc_cov_sample(DM(xnorm).full(), DM(Xnorm).full(), DM(ellopt).full(), DM(sf2opt).full())
        invKYnorm = mtimes(invK, Ynorm)
        mean = mtimes(k.T, invKYnorm)
        # mean2[i]  = mtimes(k2.T,invKY2[:,i])
        var = sf2opt - mtimes(mtimes(k.T, invK), k) + hypopt[-1]

        # meanfcn = Function('meanfcn', [x], [mean * stdY + meanY])
        # # meanfcn2 = Function('meanfcn2',[x,Xnorm2,invKY2],[mean2*stdY + meanY])
        # varfcn = Function('varfcn', [x], [var * stdY ** 2])
        # varfcnsd = Function('varfcnsd',[x],[var])
        return mean, var  # * stdY + meanY, var * stdY ** 2  # , meanfcn2, varfcnsd

    def covSEard(self, kernel='Matern32'):
        nx_dim = self.nx
        ell = SX.sym('ell', nx_dim)
        sf2 = SX.sym('sf2')
        x, z = SX.sym('x', nx_dim), SX.sym('z', nx_dim)
        if kernel == 'RBF':
            dist = sum1((x - z) ** 2 / ell)
            covSEfcn = Function('covSEfcn', [x, z, ell, sf2], [sf2 * exp(-.5 * dist)])
        elif kernel == 'Matern32':
            dist = sqrt(sum1((x - z) ** 2 / ell))
            t_1 = 1 + 3 ** 0.5 * dist
            t_2 = np.exp(-3 ** 0.5 * dist)
            cov_matrix = sf2 * t_1 * t_2
            covSEfcn = Function('covSEfcn', [x, z, ell, sf2], [cov_matrix])
        elif kernel == 'Matern52':
            dist = sqrt(sum1((x - z) ** 2 / ell))
            t_1 = 1 + 5 ** 0.5 * dist + 5 / 3 * dist ** 2
            t_2 = np.exp(-5 ** 0.5 * dist)
            cov_matrix = sf2 * t_1 * t_2
            covSEfcn = Function('covSEfcn', [x, z, ell, sf2], [cov_matrix])
        else:
            print('ERROR no kernel with name ', kernel)
            covSEfcn = Function('covSEfcn', [x, z, ell, sf2], [0.])
        return covSEfcn


class solutions:
    def __init__(self, x_opt, y_opt, maxfun, output_dict):
        x = x_opt.detach().numpy()
        self.x = x.squeeze()
        self.f = y_opt.detach().numpy()
        self.maxfun = maxfun
        self.success = 0
        self.output_dict = output_dict

    def __str__(self):
        message = '****** Bayesian Optimization using Torch Results ******' \
                  '\n Solution xmin = ' + str(self.x) + \
                  '\n Objective value f(xmin) = ' + str(self.f) + \
                  '\n With ' + str(self.maxfun) + ' Evaluations '
        return message

# ------------------------------------------------
#  This file contains extra functions to perform
#  additional operations needed everywhere
#  e.g. Create objective with penalties.
#  Generate initial points for the model-based methods
