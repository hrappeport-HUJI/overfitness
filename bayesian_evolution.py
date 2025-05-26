import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from itertools import product

if not __name__ == '__main__':
    print("Importing bayesian_evolution.py")

# --------------------------------- Linear ---------------------------------
def get_idx(n, k):
    idx = np.zeros(shape=(k, 2), dtype=int)
    for i in range(k):
        idx[i, :] = [(i + i//n)%n, i%n]
    return idx

#  Complexity based on number of non-zero entries
def sample_parameters_non_empty_entries(n, k, mean=0, std=1):
    A = np.zeros(shape=(n, n))
    idx = get_idx(n, k)
    entries = np.random.normal(mean, std, size=k)
    A[idx[:, 0], idx[:, 1]] = entries
    return A

 # Complexity based on rank
def sample_parameters_rank_1(n, k, mean=0, std=1):
    eig_vecs_left = np.random.normal(mean, std, size=(k, n))
    eig_vecs_right = np.random.normal(mean, std, size=(k, n))
    A = eig_vecs_left.T @ eig_vecs_right

    return A

def fitness_single_linear(A, E, gamma=1):
    # A is a 3X3 matrix. E is a pair of 3-vectors (shape (2,3)), indicating the signal and optimal response.
    return np.exp(-gamma * np.linalg.norm(A@E[0] - E[1]))

def fitness_population_linear(population, E, gamma=1):
    # population is a 3D array of shape (N, 3, 3). E is a pair of 3-vectors (shape (2,3))
    return np.exp(-gamma * np.linalg.norm(population@E[0].T - E[1], axis=1))

def sample_environments_linear(T, A, n, xi=0.1):
    # T is the number of environments. Returns T pairs of n-vectors (shape (T, 2,n))
    E = np.zeros((T, 2, n))
    E[:, 0, :] = np.random.randn(T, n)
    E[:, 1, :] = (A@E[:, 0, :].T).T
    E[:, 1, :] += xi * np.random.randn(T, n)
    return E

# --------------------------------- Polynomial ---------------------------------
def sample_parameters_polynomial(n, k, mean=0, std=1):
    # here k is the degree of the polynomial and A is the coefficients of the polynomial
    A = np.zeros(n)
    A[:k] = np.random.normal(mean, std, size=k)
    return A

def fitness_single_polynomial(A, E, gamma=1):
    # A is a n-vector. E is a pair of n-vectors (shape (2,n)), indicating the signal and optimal response.
    V = np.vander(E[0], increasing=True)
    return np.exp(-gamma * np.linalg.norm(V@A - E[1]))

def fitness_population_polynomial(population, E, gamma=1):
    # population is a 3D array of shape (N, n). E is a pair of n-vectors (shape (2,n))
    V = np.vander(E[0], increasing=True)
    # Reshape E[1] to (n, 1) to allow broadcasting
    E_target = E[1].reshape(-1, 1)
    # Calculate fitness for each individual in the population
    fitness_values = np.exp(-gamma * np.linalg.norm(V @ population.T - E_target, axis=0))
    return fitness_values

def sample_environments_polynomial(T, A, n, xi=0.1):
    # T is the number of environments. Returns T pairs of n-vectors (shape (T, 2,n))
    E = np.zeros((T, 2, n))
    E[:, 0, :] = np.random.randn(T, n)
    E[:, 1, :] = np.polyval(A[::-1], E[:, 0, :].T).T  # A[::-1] to match the order of coefficients in np.polyval to the vandermonde matrix
    # add noise to the target response
    E[:, 1, :] += xi * np.random.randn(T, n)
    return E


def set_function_type(setting):
    global sample_parameters, fitness_single, fitness_population, sample_environments
    if setting == "polynomial":
        sample_parameters = sample_parameters_polynomial
        fitness_single = fitness_single_polynomial
        fitness_population = fitness_population_polynomial
        sample_environments = sample_environments_polynomial
    elif setting == "linear":
        sample_parameters = sample_parameters_non_empty_entries
        fitness_single = fitness_single_linear
        fitness_population = fitness_population_linear
        sample_environments = sample_environments_linear

def sample_parameters_population(n, ks, class_size, gt=None):
    shape = sample_parameters(n, ks[0]).shape
    population_M = np.zeros(shape=(len(ks) * class_size, *shape)) if gt is None else np.zeros(shape=((len(ks) * class_size)+1, *shape))
    for i, k in enumerate(ks):
        for j in range(class_size):
            population_M[i * class_size + j, :] = sample_parameters(n, k)
    if gt is not None:
        population_M[-1, :] = gt
    return population_M

class ExperimentLogger:
    def __init__(self, exp_name, exp_parameters, exp_log_items=None):
        self.exp_name = exp_name
        self.exp_parameters = exp_parameters
        self.data = {}
        T = exp_parameters["T"]
        if exp_log_items is not None:
            for item in exp_log_items:
                shape = exp_log_items[item]["shape"]
                dtype = exp_log_items[item]["dtype"]
                self.data[item] = np.zeros(shape=(T, *shape), dtype=dtype)

    def log_iteration(self, data, t):
        for item in data:
            self.data[item][t] = data[item]

    def close(self, to_compute):
        if "last_iter_class_freqs" in to_compute:
            last_iter_class_freqs = np.sum(self.data["frequency"][-1].reshape((len(self.exp_parameters["ks"]), self.exp_parameters["class_size"])), axis=1)
            self.data["last_iter_class_freqs"] = last_iter_class_freqs

    def save(self, to_save="all"):
        if to_save == "all":
            save_obj = self
        else:
            save_obj = ExperimentLogger(self.exp_name, self.exp_parameters)
            for item in to_save:
                save_obj.data[item] = self.data[item]
        with open(f"experiments/{self.exp_name}.pkl", "wb") as f:
            pickle.dump(save_obj, f)

def run(exp_name, true_k, fitness_gamma=0.05, to_save=["last_iter_class_freqs"], n=3, T=1000, ks=None,
        class_size=2000, env_switches=None, n_envs=None, env_change_rate=None, xi=0.1, to_return=None,
        insert_gt=False, adaptive_classes=False):
    """
    Runs the experiment for T time steps.
    :param exp_name: Name of the experiment (used for saving)
    :param true_k: The true complexity of the environment
    :param fitness_gamma: The selection strength
    :param to_save: List of items to save
    :param n: The size of the matrix
    :param T: The number of time steps
    :param ks: The complexity classes participating
    :param class_size: The number of matrices in each class
    :param env_switches: The time steps at which the environment changes
    :param n_envs: The number of environments, if env_switches is not provided
    :param env_change_rate: The rate at which the environment changes (Poisson process), if env_switches is not provided
    :param xi: The noise level in the environment
    :param to_return: List of items to return

    If to_return is None - Does not return anything, but saves the experiment to a file with the name exp_name.
    If to_return is a list of items, returns a dictionary with the items as keys and the corresponding data as values.
    """
    if env_switches is None:  # can provide env_switches directly
        if n_envs is not None:  # if n_envs is provided, divide T into n_envs equal parts
            env_switches = [int(T / (n_envs) * i) for i in range(n_envs)] + [T]
        elif env_change_rate is not None:  # if env_change_rate is provided, randomly switch environments
            env_idxs = np.zeros(T, dtype=int)
            env_idxs[0] = np.random.randint(n_envs)
            env_switches = [0]
            for t in range(1, T):
                env_idxs[t] = env_idxs[t - 1] if np.random.rand() > env_change_rate else np.random.randint(n_envs)
                if env_idxs[t] != env_idxs[t - 1]:
                    env_switches.append(t)
            env_switches.append(T)
        else:
            raise ValueError("Either n_envs or env_change_rate must be provided")
    n_types = len(ks) * class_size if not insert_gt else len(ks) * class_size + 1
    exp_logger = ExperimentLogger(exp_name,
                              exp_parameters={
                                  "T": T, "n": n, "ks": ks, "class_size": class_size,
                                  "n_types": n_types, "true_k": true_k,
                                  "fitness_gamma": fitness_gamma, "env_switches": env_switches,
                                  "env_change_rate": env_change_rate},
                              exp_log_items={
                                 **({"frequency": {"shape": (n_types,), "dtype": float}} if not adaptive_classes else {}),
                                 "class_frequency": {"shape": (len(ks) if not insert_gt else len(ks)+1,), "dtype": float},
                                 "fitness": {"shape": (n_types,), "dtype": float}})

    E = np.zeros((T, 2, n))
    env_slices = [slice(env_switches[i], env_switches[i + 1]) for i in range(len(env_switches) - 1)]
    for env_slice in env_slices:
        env_M = sample_parameters(n, true_k)
        E[env_slice, :, :] = sample_environments(env_slice.stop - env_slice.start, env_M, n, xi)

    population_M = sample_parameters_population(n, ks, class_size) if not insert_gt else\
        sample_parameters_population(n, ks, class_size, gt=env_M)
    frequency = np.ones(n_types) / n_types

    if not adaptive_classes:
        for t in range(T):
            fitness = fitness_population(population_M, E[t], fitness_gamma)
            class_freqs = np.sum(frequency.reshape((len(ks), class_size)), axis=1) if not insert_gt else \
                np.concatenate([np.sum(frequency[:-1].reshape((len(ks), class_size)), axis=1), np.array([frequency[-1]])])
            exp_logger.log_iteration(
            {
                "frequency": frequency,
                "class_frequency": class_freqs,
                "fitness": fitness
            }, t=t)
            frequency = frequency * fitness
            frequency = frequency / np.sum(frequency)  # normalize

    else:
        class_freqs = np.ones(len(ks)) / len(ks)
        for t in range(T):
            fitness = fitness_population(population_M, E[t], fitness_gamma)
            exp_logger.log_iteration(
            {
                "class_frequency": class_freqs,
                "fitness": fitness
            }, t=t)
            # class fitness is max fitness of the class
            class_fitness = np.zeros(len(ks))
            for i, k in enumerate(ks):
                class_fitness[i] = np.max(fitness[i*class_size:(i+1)*class_size])
            class_freqs = class_freqs * class_fitness
            class_freqs = class_freqs / np.sum(class_freqs)  # normalize

    # exp_logger.close(to_compute=["last_iter_class_freqs"])
    if to_save is not None:
        exp_logger.save(to_save=to_save)

    if to_return is not None:
        rv = {}
        for item in to_return:
            rv[item] = exp_logger.data[item]
        return rv


def plot_bubble_chart(M):
    sns.set_style("whitegrid")
    sns.set_context("talk")
    plt.rcParams.update(
        {'axes.labelsize': 28, 'axes.labelweight': 'normal', 'ytick.labelsize': 28, 'xtick.labelsize': 28})

    n_q_stars, n_bins = M.shape  # should be (9, 9)

    # Create a figure
    fig, ax = plt.subplots(figsize=(7, 6))

    # We want x-axis for q_star = 1..9, y-axis for bin index = 1..9
    # and bubble size ~ M[i, j].
    for i in range(n_q_stars):
        arg_max_i = np.argmax(M[i])
        for j in range(n_bins):
            # Scale the marker size so that it's visually distinguishable.
            # You can tweak the scaling factor to taste (e.g. 500, 1000, etc.).
            bubble_size = M[i, j] * 1000

            # scatter at (q_star, bin) = (i+1, j+1)
            color = 'steelblue' if j == arg_max_i else 'lightgray'
            ax.scatter(
                i + 1, j + 1,
                s=bubble_size,
                color=color,
                alpha=0.6,
                edgecolors='black'
            )

    # Make the axis ticks go from 1 to 9
    ax.set_xticks(np.arange(1, n_q_stars + 1))
    ax.set_yticks(np.arange(1, n_bins + 1))

    plt.tight_layout()
    return ax


def plot_muller(matrix, ax, type_freqs=None, type_colors=None):
    """
    Plots a Muller plot given an input matrix of shape (population_size, T)
    where each column sums to unity.
    """
    T, population_size  = matrix.shape
    if not np.allclose(matrix.sum(axis=1), np.ones(T)):
        raise ValueError("Each row of the input matrix must sum to unity.")
    y_stack = np.cumsum(matrix, axis=1)

    if type_colors is not None:
        color = type_colors[0]
        ax.fill_between(range(T), 0, y_stack[:, 0], step='mid', color=color)
        for i in range(1, population_size):
            color = type_colors[i]
            ax.fill_between(range(T), y_stack[:, i-1], y_stack[:, i], step='mid', color=color)
    else:
        ax.fill_between(range(T), 0, y_stack[:, 0], step='mid')
        for i in range(1, population_size):
            ax.fill_between(range(T), y_stack[:, i-1], y_stack[:, i], step='mid')

    # plot type frequencies
    if type_freqs is not None:
        ax.plot(np.cumsum(type_freqs[:-1, :], axis=1),  ls="--", color='k', linewidth=5)
        # at time 0, plot the type frequencies using the colors as a fill_between from -10 to 0
        y_1, y_2 = 0, 0
        if type_colors is None:
            type_colors = plt.cm.viridis(np.linspace(0, 1, len(type_freqs)))
        for type_i, type_color in enumerate(type_colors):
          y_2 += type_freqs[0, type_i]
          ax.fill_between(np.arange(-T//20, 0.01), y_1, y_2, color=type_color, alpha=0.5)
          y_1 += type_freqs[0, type_i]

    ax.set_xlabel('Generations')
    ax.set_ylabel('Frequency')
