import logging

import matplotlib.pyplot as plt
import numpy as np

from noise import *


def float_eq(x, y):
    return abs(x - y) < 0.000001


def test_hoeffding():
    a = 1.
    k = 10000
    beta = 0.01
    tau = hoeffding_bound_inverse(0.01, k, a)
    beta_ = hoeffding_bound(tau, k, a)
    assert float_eq(beta, beta_)


def test_bernstein():
    a = 1.
    k = 10000
    beta = 0.01
    var = 2.
    tau = bernstein_bound_inverse(0.01, k, a, var)
    beta_ = bernstein_bound(tau, k, a, var)
    assert float_eq(beta, beta_)


def test_sub_exponential():
    k = 10000
    beta = 0.01
    nu = 2.0
    alpha = 2.0
    tau = sub_exponential_bound_inverse(0.01, k, nu, alpha)
    beta_ = sub_exponential_bound(tau, k, nu, alpha)
    assert float_eq(beta, beta_)


def test_gen_large():
    lap = LaplaceNoise(sensitivity=1., epsilon=1., seed=0)
    large = lap.generate_large(10)
    assert large >= 10


def test_lap():
    lap = LaplaceNoise(sensitivity=1., epsilon=1., seed=0)
    assert lap.variance() == 2.
    beta = 0.01
    tau = lap.tail_bound_inverse(beta)
    beta_ = lap.tail_bound(tau)
    assert float_eq(beta, beta_)


def test_geom():
    geom = GeometricNoise(sensitivity=1., epsilon=1., seed=0)
    assert float_eq(geom.variance(), 2. * np.e / ((np.e - 1) ** 2))
    tau = 5.5
    beta = geom.tail_bound(tau)
    tau_ = geom.tail_bound_inverse(beta)
    assert float_eq(tau_, np.ceil(tau))


def test_gauss():
    gauss = GaussianNoise(l2sensitivity=1., epsilon=1., seed=0)
    beta = 0.01
    tau = gauss.tail_bound_inverse(beta)
    beta_ = gauss.tail_bound(tau)
    assert float_eq(beta, beta_)


def test_bounds():
    n = 100000
    lap = LaplaceNoise(sensitivity=1., epsilon=1., seed=0)
    geom = GeometricNoise(sensitivity=1., epsilon=1., seed=0)
    gauss = GaussianNoise(l2sensitivity=1., epsilon=1., seed=0)
    lap_noises = lap.generate(n)
    lap_max = max(np.abs(lap_noises))
    assert lap_max < lap.max_noise(0.1, n)
    geom_noises = geom.generate(n)
    geom_max = max(np.abs(geom_noises))
    assert geom_max <= geom.max_noise(0.1, n)
    gauss_noises = gauss.generate(n)
    gauss_max = max(np.abs(gauss_noises))
    assert gauss_max <= gauss.max_noise(0.1, n)
    xrng = np.arange(0.001, 0.5, 0.001)
    plt.plot(xrng, [lap_max] * len(xrng), label="Laplace empirical max")
    plt.plot(xrng, [geom_max] * len(xrng), label="Geometric empirical max")
    plt.plot(xrng, [gauss_max] * len(xrng), label="Gaussian empirical max")
    plt.plot(xrng, lap.max_noise(xrng, n), label="Laplace bound")
    plt.plot(xrng, geom.max_noise(xrng, n), label="Geometric bound")
    plt.plot(xrng, gauss.max_noise(xrng, n), label="Gaussian bound")
    plt.legend(loc='best')
    plt.show()


def test_compare_noise():
    n = 100000
    lap = LaplaceNoise(sensitivity=1., epsilon=1., seed=0)
    geom = GeometricNoise(sensitivity=1., epsilon=1., seed=0)
    gauss = GaussianNoise(l2sensitivity=1., epsilon=1., seed=0)
    lap_noises = lap.generate(n)
    lap_max = max(np.abs(lap_noises))
    assert lap_max < lap.max_noise(0.1, n)
    geom_noises = geom.generate(n)
    geom_max = max(np.abs(geom_noises))
    assert geom_max <= geom.max_noise(0.1, n)
    gauss_noises = gauss.generate(n)
    gauss_max = max(np.abs(gauss_noises))
    assert gauss_max <= gauss.max_noise(0.1, n)
    xrng = np.arange(0.001, 0.5, 0.001)
    plt.plot(xrng, [lap_max] * len(xrng), label="Laplace empirical max")
    plt.plot(xrng, [geom_max] * len(xrng), label="Geometric empirical max")
    plt.plot(xrng, [gauss_max] * len(xrng), label="Gaussian empirical max")
    plt.plot(xrng, lap.max_noise(xrng, n), label="Laplace bound")
    plt.plot(xrng, geom.max_noise(xrng, n), label="Geometric bound")
    plt.plot(xrng, gauss.max_noise(xrng, n), label="Gaussian bound")
    plt.legend(loc='best')
    plt.show()


def test_compare_sums():
    n = 100000
    k = 100
    lap = LaplaceNoise(sensitivity=1., epsilon=1., seed=0)
    geom = GeometricNoise(sensitivity=1., epsilon=1., seed=0)
    gauss = GaussianNoise(l2sensitivity=1., epsilon=1., seed=0)
    lap_noises = np.zeros(n)
    geom_noises = np.zeros(n)
    gauss_noises = np.zeros(n)
    for i in range(k):
        lap_noises = np.add(lap_noises, lap.generate(n))
        geom_noises = np.add(geom_noises, geom.generate(n))
        gauss_noises = np.add(gauss_noises, gauss.generate(n))

    lap_max = max(np.abs(lap_noises))
    assert lap_max < lap.max_sum_noise(0.1, n, k)
    geom_max = max(np.abs(geom_noises))
    assert geom_max < geom.max_sum_noise(0.1, n, k)
    gauss_max = max(np.abs(gauss_noises))
    assert gauss_max < gauss.max_sum_noise(0.1, n, k)

    xrng = np.arange(0.001, 0.5, 0.001)
    plt.plot(xrng, [lap_max] * len(xrng), label="Laplace empirical max")
    plt.plot(xrng, [geom_max] * len(xrng), label="Geometric empirical max")
    plt.plot(xrng, [gauss_max] * len(xrng), label="Gaussian empirical max")
    plt.plot(xrng, lap.max_sum_noise(xrng, n, k), label="Laplace bound")
    plt.plot(xrng, geom.max_sum_noise(xrng, n, k), label="Geometric bound")
    plt.plot(xrng, gauss.max_sum_noise(xrng, n, k), label="Gaussian bound")
    plt.legend(loc='best')
    plt.show()
