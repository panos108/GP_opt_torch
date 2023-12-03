# SI-Bayesian-Optimization
This repo aims to solve Bayesian optimization using [Pyro](https://pyro.ai/) and [pytorch simple](https://pytorch.org/). The objective function and the constraints can be defined in numpy format. The repository implements the following: 

**Unconstrained Optimization**

The unconstrained optimization can minimize one of the following acquisition functions: mean, Lower confidence bound or negative expected improvement. 
The default solver here is BDGS via Pyro. 

**Constrained Optimization**

The constrained optimization can minimize one of the following acquisition functions: mean, Lower confidence bound or negative expected improvement. The constraints can be satisfied with respect to the mean (*probabilistic to be included*)
[Casadi](https://web.casadi.org/) is used and ipopt. 

## Installation

```bash
git clone https://git.illumina.com/ProductDevelopment/SI-Bayesian-Optimization.git
```
Additional packages needed 
```bash
pip install casadi 
pip3 install pyro-ppl
pip install sobol_seq
pip install pyDOE
```

## Options for solver
 The value depited is the default one.
 
***objective***           *Objective to be minimized, if it is the real experiment give Nan*
 
***xo***                      *initial point. It is not required*

***bounds (REQUIRED)***       *Bounds for the decision variable*
 
***maxfun=20***                 *Number of iterations*
 
***N_initial=4***                *Number of initial points*
 
***select_kernel='Matern52'***    *Kernel for Gaussian process*

***acquisition='LCB'***           *Acquisition function*

***casadi=False***                *Solve the problem via casadi and ipopt (this is used for constrained problems*

***constraints = None***          *No constraints by defaults*

***probabilistic=False***        *To be implemented for probabilistic constraints*

***print_iteration=False***       *Print iterations*


## Example give data points but from real experiment
```python
from Bayesian_opt_Pyro.utilities_full import BayesOpt
assert pyro.__version__.startswith('1.5.1')
pyro.enable_validation(True)  # can help with debugging
pyro.set_rng_seed(1)


def f1(x):
    return (6 * x[0] - 2)**2 * np.sin(12 * x[1] - 4) + 100*max(0,(6 * x[0] - 2)**2  - 1)**2
def g1(x):
    return (6 * x[0] - 2)**2  - 1


def fun_eval(x):
    return f1(x), [-g1(x)]

lower = np.array([0.0]*2)
upper = np.array([1.]*2)

X = np.random.rand(4,2)
Y = np.array([[f1(X[i]) for i in range(4)], [-g1(X[i]) for i in range(4)]]).T


solution_noconstr = BayesOpt().solve(None, xo=lower, bounds=(lower,upper),
                             maxfun=20, constraints=0, casadi=True, print_iteration=True,
                             X_initial=X, Y_initial=Y, constraints=1,casadi=True)
```
## If function is available to be evaluated
```
python
def f1(x):
    return (6 * x[0] - 2)**2 * np.sin(12 * x[1] - 4) + 100*max(0,(6 * x[0] - 2)**2  - 1)**2
def g1(x):
    return (6 * x[0] - 2)**2  - 1


def fun_eval(x):
    return f1(x), [-g1(x)]

lower = np.array([0.0]*2)
upper = np.array([1.]*2)

solution_noconstr = BayesOpt().solve(fun_eval, xo=lower, bounds=(lower,upper),
                             maxfun=20, constraints=0, casadi=True, print_iteration=True,
                             constraints=1,casadi=True)


print(solution1)
```

