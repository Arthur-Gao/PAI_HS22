import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, Sum, Product
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.special import erf
from scipy.stats import norm
domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
SEED = 0

""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """
        
        self.v_min = 1.2
        fk = Matern(length_scale=0.5, nu=2.5,length_scale_bounds="fixed")
        fkvar = ConstantKernel(constant_value = 0.5, constant_value_bounds="fixed")
        fnoise = WhiteKernel(noise_level=0.15)
        self.fk = Sum(Product(fk, fkvar),fnoise)
        self.f_gp = GaussianProcessRegressor(kernel=self.fk, alpha=1e-10, normalize_y=False, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10)

        
        vk = Matern(length_scale=0.5, nu=2.5)
        vkvar = ConstantKernel(constant_value = np.sqrt(2), constant_value_bounds="fixed")
        vkmean = ConstantKernel(constant_value = 1.5)
        vnoise = WhiteKernel(noise_level=0.0001)
        self.vk = Sum(Sum(vkmean, Product(vk, vkvar)),vnoise)
        self.v_gp = GaussianProcessRegressor(kernel=self.vk, alpha=1e-10, normalize_y=False, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10)

        self.results = np.zeros((1, 3))
        self.iter_count = 0
    
        # TODO: enter your code here


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        # return np.atleast_2d(self.optimize_acquisition_function())
        return self.optimize_acquisition_function()
        


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # # TODO: enter your code here
        # raise NotImplementedError
        f_preds = self.f_gp.predict(np.atleast_2d(x), return_std=True)
        v_preds = self.v_gp.predict(np.atleast_2d(x), return_std=True)
        trade_off = 0.01 # 0.9 or 1.96 or 0.05 or 0.01
        gamma_v = (v_preds[0] - (self.v_min+ trade_off)) / v_preds[1]
        pi_v = norm.cdf(gamma_v)

        if (pi_v < 0.5):
            #reshape to fit the correct shape
            return np.reshape(pi_v, [1])
        else:
            # if ProbabilityOfImprovement of V is higher than 1/2, compute the ExpectedImprovement of function F and combine results
            optimal_x_idx = np.argmax(self.results[:, 1])
            best_f = self.results[optimal_x_idx, 1]
            gamma_f = (f_preds[0] - (best_f + trade_off)) / f_preds[1]
            # Compute EI based on some temporary variables that can be reused in
            # gradient computation
            tmp_erf = erf(gamma_f / np.sqrt(2))
            tmp_ei_no_std = 0.5 * gamma_f * (1 + tmp_erf) \
                            + np.exp(-gamma_f ** 2 / 2) / np.sqrt(2 * np.pi)
            ei_f = f_preds[1] * tmp_ei_no_std
            #reshape to fit the correct shape
            return np.reshape(pi_v*ei_f, [1])
        # f= self.f_gp.predict(np.atleast_2d(x), return_std=True)
        # v = self.v_gp.predict(np.atleast_2d(x), return_std=True)
        # beta = 1.5
        # max_f = np.amax(f, axis=1)[1]
        # ucb = (f[0] - max_f) + f[0]*beta

        # return ucb[0]


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """
        # x = x[0,0]
        # print(np.hstack((x, f, v)))
        # print(x)
        # print(f)
        # print(v)
        tmp = np.zeros((1,3))
        if self.iter_count == 0:
            self.results[self.iter_count, :] = np.hstack((x, f, v))
        else:
            # print(tmp.shape)
            # print(self.results.shape)
            tmp[0, :] = np.hstack((x[0,0], f, v))
            self.results = np.concatenate((self.results, tmp), axis=0)
        self.iter_count += 1
        x_train = np.transpose(np.atleast_2d(self.results[:self.iter_count, 0]))
        f_train = np.transpose(np.atleast_2d(self.results[:self.iter_count, 1]))
        v_train = np.transpose(np.atleast_2d(self.results[:self.iter_count, 2]))

        # Fit GPRs
        
        self.f_gp.fit(x_train, f_train)
        self.v_gp.fit(x_train, v_train)

        # TODO: enter your code here
        # raise NotImplementedError

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
        constraint_idx = np.where(self.results[:,2] >= self.v_min)[0]
        satisfied_constraint = self.results[constraint_idx, :]
        print(satisfied_constraint)
        print(constraint_idx)
        print(self.results)
        return satisfied_constraint[np.argmax(satisfied_constraint[:,1]), 0]

        # TODO: enter your code here
        # raise NotImplementedError


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0

def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val > SAFETY_THRESHOLD]
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init



def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)


    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)
        print("kkkkkkkkkk")

    print("outof loop")
    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()