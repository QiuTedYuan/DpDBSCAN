from abc import ABC, abstractmethod

import numpy as np
import sklearn.utils
from nptyping import NDArray, Shape, typing_

Noises = NDArray[Shape["N"], typing_.Number]


# pr[ |sum| > t ] < ?, conditioned on |x_i| < a
def hoeffding_bound(t, k: int, a: float):
    t2 = np.power(t, 2)
    a2 = a ** 2
    return 2 * np.exp(- t2 / (2 * k * a2))


def hoeffding_bound_inverse(beta, k: int, a: float):
    return np.sqrt(2 * k * np.log(2. / beta)) * a


def chebyshev_bound(t, k: int, var: float):
    return k * var / np.power(t, 2)


def chebyshev_bound_inverse(beta, k: int, var: float):
    return np.sqrt(k * var / beta)


def bernstein_bound(t, k: int, a: float, var: float):
    t2 = np.power(t, 2)
    return 2 * np.exp(- t2 / 2. / (k * var + a * t / 3.))


def bernstein_bound_inverse(beta, k: int, a: float, var: float):
    ln = np.log(2. / beta)
    ln2 = np.power(ln, 2)
    a2 = a ** 2
    return a / 3. * ln + np.sqrt(a2 * ln2 / 9. + 2 * k * var * ln)


# sum of SE(nu^2, alpha) vars denotes by SE(k*nu^2, alpha)
def sub_exponential_bound(t, k: int, nu: float, alpha: float):
    t2 = np.power(t, 2)
    nu2 = k * (nu ** 2)
    return 2 * np.exp(- 1. / 2. * np.minimum(t2 / nu2, t / alpha))


def sub_exponential_bound_inverse(beta, k: int, nu: float, alpha: float):
    ln = 2. * np.log(2. / beta)
    t1 = np.sqrt(k * ln) * nu
    t2 = ln * alpha
    return np.maximum(t1, t2)


class NoiseGenerator(ABC):

    @abstractmethod
    def generate(self, size: int) -> Noises:
        pass

    @abstractmethod
    def variance(self):
        pass

    @abstractmethod
    def tail_bound(self, t):
        pass

    @abstractmethod
    def tail_bound_inverse(self, beta) -> float:
        pass

    def max_noise(self, beta, n: int):
        return self.tail_bound_inverse(beta / n)

    def sum_trivial(self, beta, n: int, k: int):
        return k * self.max_noise(beta, n)

    def sum_hoeffding(self, beta, n: int, k: int):
        a = self.max_noise(beta / 2., n)
        return hoeffding_bound_inverse(beta / 2. / n, k, a)

    def sum_chebyshev(self, beta, n: int, k: int):
        return chebyshev_bound_inverse(beta / n, k, self.variance())

    def sum_bernstein(self, beta, n: int, k: int):
        a = self.max_noise(beta / 2., n)
        return bernstein_bound_inverse(beta / 2. / n, k, a, self.variance())

    def max_sum_noise(self, beta, n: int, k: int):
        res = self.sum_trivial(beta, n, k)
        res = np.minimum(res, self.sum_hoeffding(beta, n, k))
        return res


class SubExponentialNoise(NoiseGenerator, ABC):
    def __init__(self, exp_nu: float, exp_alpha: float):
        self.exp_nu = exp_nu
        self.exp_alpha = exp_alpha

    def sum_sub_exponential(self, beta, n: int, k: int):
        return sub_exponential_bound_inverse(beta / n, k, self.exp_nu, self.exp_alpha)

    def max_sum_noise(self, beta, n: int, k: int):
        res = super().max_sum_noise(beta, n, k)
        res = np.minimum(res, self.sum_sub_exponential(beta, n, k))
        return res


class LaplaceNoise(SubExponentialNoise):
    def __init__(self, seed, sensitivity: float, epsilon: float):
        self.random_state: np.random.RandomState = sklearn.utils.check_random_state(seed)
        self.epsilon: float = epsilon
        self.sensitivity: float = sensitivity
        self.epsilon_: float = 1. * epsilon / sensitivity
        super().__init__(2. / self.epsilon_, np.sqrt(2.) / self.epsilon_)

    def generate(self, size: int) -> Noises:
        return self.random_state.laplace(0, 1. / self.epsilon_, size)

    def variance(self):
        return 2 / (self.epsilon_ ** 2)

    def tail_bound(self, t):
        return np.exp(-self.epsilon_ * t)

    def tail_bound_inverse(self, beta) -> float:
        return 1. / self.epsilon_ * np.log(1. / beta)


class GeometricNoise(SubExponentialNoise):
    def __init__(self, seed, sensitivity: float, epsilon: float):
        self.random_state: np.random.RandomState = sklearn.utils.check_random_state(seed)
        self.epsilon: float = epsilon
        self.sensitivity: float = sensitivity
        self.epsilon_: float = 1. * epsilon / sensitivity
        self.alpha = np.exp(self.epsilon_)
        super().__init__(8 * self.alpha / ((self.alpha - 1) ** 2),
                         min(0.5, np.sqrt(2 * np.log((self.alpha + 1) ** 2 / 4. / self.alpha))))

    def __zero__(self, size):
        zero_prob = (self.alpha - 1.) / (self.alpha + 1.)
        rnd = self.random_state.random(size)
        return np.where(rnd < zero_prob, 0, 1)

    def __sign__(self, size):
        rnd = self.random_state.random(size)
        return np.where(rnd < 0.5, -1, 1)

    def __geom__(self, size):
        return self.random_state.geometric(1. - 1. / self.alpha, size)

    def generate(self, size: int) -> Noises:
        return self.__zero__(size) * self.__sign__(size) * self.__geom__(size)

    def variance(self):
        return 2. * self.alpha / ((self.alpha - 1.) ** 2)

    # Pr[|Z| >= t] = Pr[|Z| >= ceil(t)]
    def tail_bound(self, t) -> float:
        return 2. * self.alpha / (self.alpha + 1) * np.exp(-self.epsilon_ * np.ceil(t))

    # Pr[|Z| >= floor(t)] = Pr[|Z| >= t] = beta
    def tail_bound_inverse(self, beta) -> int:
        return np.floor(1. / self.epsilon_ * np.log(2. * self.alpha / (self.alpha + 1) / beta))

    def sum_geometric(self, beta, n: int, k: int):
        beta_ = beta / n
        threshold = self.alpha * np.log(2. / beta_)
        return (4. * np.sqrt(k) / (self.alpha - 1.) * np.sqrt(threshold)
                * np.where(k >= threshold, 1., np.sqrt(threshold)))

    def max_sum_noise(self, beta, n: int, k: int):
        res = super().max_sum_noise(beta, n, k)
        res = np.minimum(res, self.sum_geometric(beta, n, k))
        return res


class GaussianNoise(NoiseGenerator):
    def __init__(self, seed, l2sensitivity: float, epsilon, delta=10 ** -5):
        self.random_state: np.random.RandomState = sklearn.utils.check_random_state(seed)
        self.epsilon: float = epsilon
        self.delta: float = delta
        self.sigma = np.sqrt(2. * np.log(1.25 / delta) * l2sensitivity / epsilon)

    def generate(self, size: int) -> Noises:
        return self.random_state.normal(0, self.sigma, size)

    def variance(self):
        return self.sigma ** 2

    def tail_bound(self, t) -> float:
        return 2 * np.exp(-np.power(t, 2) / 2. / self.variance())

    def tail_bound_inverse(self, beta) -> float:
        return np.sqrt(2. * np.log(2. / beta)) * self.sigma

    def max_sum_noise(self, beta, n: int, k: int):
        res = super().max_sum_noise(beta, n, k)
        res = np.minimum(res, np.sqrt(2. * np.log(2. * n / beta)) * self.sigma * np.sqrt(k))
        return res
