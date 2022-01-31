#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import genfromtxt

__authors__ = ["Baptiste Coudray", "Quentin Berthet", "Alexandre Rosset", "Jorge Goncalves"]
__date__ = "23.03.2020"
__course__ = "MTI"
__description__ = "Differential Equations and SEIRD Model"


def explicit_euler_method(f, y_n, t_n, dt):
    return y_n + dt * f(y_n, t_n)


def midpoint_method(f, y_n, t_n, dt):
    v_prime = y_n + (dt / 2) * f(y_n, t_n)
    return y_n + dt * f(v_prime, t_n + (dt / 2))


def order_4_runge_kutta(f, y_n, t_n, dt):
    k1 = dt * f(y_n, t_n)
    k2 = dt * f(y_n + (k1 / 2), t_n + (dt / 2))
    k3 = dt * f(y_n + (k2 / 2), t_n + (dt / 2))
    k4 = dt * f(y_n + k3, t_n + dt)
    return y_n + (1 / 6) * (k1 + 2 * (k2 + k3) + k4)


def single_plot_3d(v, label):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(v[:, 0], v[:, 1], v[:, 2], label=label)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{label}.png", bbox_inches="tight")
    plt.show()


def multiple_plot_3d(v1, v2, v3, label1, label2, label3):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(v1[:, 0], v1[:, 1], v1[:, 2], label=label1)
    ax.plot(v2[:, 0], v2[:, 1], v2[:, 2], label=label2)
    ax.plot(v3[:, 0], v3[:, 1], v3[:, 2], label=label3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{label1}_{label2}_{label3}.png", bbox_inches="tight")
    plt.show()


def lorenz(v, t):
    sigma = 10
    rho = 28
    beta = 8 / 3
    return np.array([sigma * (v[1] - v[0]), v[0] * (rho - v[2]) - v[1], v[1] * v[0] - beta * v[2]])


R0 = 4.2
T_INF = 3
T_INC = 5
N = 8_600_000
I0 = 1
E0 = 60
T_DEATH = 0

S, E, I, R, D = 0, 1, 2, 3, 4

REAL_INFECTED = genfromtxt("infected.csv")
REAL_RECOVERED = genfromtxt("recovered.csv")
REAL_DEATHS = genfromtxt("deaths.csv")


def seir_model(v, t):
    return np.array([
        (-R0 / T_INF) * v[I] * v[S] / N,  # S
        (R0 / T_INF) * v[I] * (v[S] / N) - (1 / T_INC) * v[E],  # E
        (1 / T_INC) * v[E] - (1 / T_INF) * v[I],  # I
        (1 / T_INF) * v[I]  # R
    ])


# https://medium.com/@djconnel_14663/seird-model-of-covid-19-596e6754c2c4
def seird_model(v, t):
    return np.array([
        (-R0 / T_INF) * v[I] * v[S] / N,  # S
        (R0 / T_INF) * v[I] * (v[S] / N) - (1 / T_INC) * v[E],  # E
        (1 / T_INC) * v[E] - (1 / T_INF) * v[I],  # I
        (1 / T_INF) * v[I] * (1 - T_DEATH),  # R
        (1 / T_INF) * v[I] * T_DEATH  # D
    ])


def main():
    vs1 = np.array([[-0.9101673912, -1.922121396, 18.18952097]])
    vs2 = np.array([[-0.9101673912, -1.922121396, 18.18952097]])
    vs3 = np.array([[-0.9101673912, -1.922121396, 18.18952097]])
    dt = 0.001
    t_max = 1.5586522 + dt
    for t in np.arange(0, t_max, dt):
        vs1 = np.append(vs1, [explicit_euler_method(lorenz, vs1[-1], t, dt)], axis=0)
        vs2 = np.append(vs3, [midpoint_method(lorenz, vs2[-1], t, dt)], axis=0)
        vs3 = np.append(vs3, [order_4_runge_kutta(lorenz, vs3[-1], t, dt)], axis=0)

    single_plot_3d(vs1, "Explicit Euler Method")
    single_plot_3d(vs2, "Midpoint Method")
    single_plot_3d(vs3, "Order 4 Runge-Kutta")
    multiple_plot_3d(vs1, vs2, vs3, "Explicit Euler Method", "Midpoint Method", "Order 4 Runge-Kutta")

    vs1 = np.array([[
        N - I0,  # S
        E0,  # E
        I0,  # I
        0,  # R
        0,  # D
    ]])

    dt = 1
    t_max_irl = 53
    t_max_simulation = 365
    global R0
    for t in np.arange(1, t_max_simulation, dt):
        vs1 = np.append(vs1, [order_4_runge_kutta(seird_model, vs1[-1], t, dt)], axis=0)
        if t == 9:
            global T_DEATH
            T_DEATH = 0.005
        # Confinement
        if t == 22:
            R0 = 3
        elif 20 < t < 40:
            R0 -= R0 * 0.05

    ts_irl = np.arange(0, t_max_irl, dt)
    ts_simulation = np.arange(0, t_max_simulation, dt)
    plt.figure()
    plt.semilogy(ts_simulation, vs1[:, 0], label="Susceptible")
    plt.semilogy(ts_simulation, vs1[:, 1], label="Exposed")
    plt.semilogy(ts_simulation, vs1[:, 2], label="Infected")
    plt.semilogy(ts_irl, REAL_INFECTED, label="Real Infected")
    plt.semilogy(ts_simulation, vs1[:, 3], label="Recovered")
    plt.semilogy(ts_irl, REAL_RECOVERED, label="Real Recovered")
    plt.semilogy(ts_simulation, vs1[:, 4], label="Deaths")
    plt.semilogy(ts_irl, REAL_DEATHS, label="Real Deaths")
    plt.xlabel("Jour")
    plt.legend()
    plt.tight_layout()
    plt.savefig("seirm.png", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
