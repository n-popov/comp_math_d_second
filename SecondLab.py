import numpy as np
from math import exp
from math import sqrt


def k_left(x):
    return x * x + 1


def q(x):
    return exp(-x)


def f_left(x):
    return 1


def k_right(x):
    return x


def f_right(x):
    return x * x


grid = [i / 10 for i in range(11)]
x_0 = 0.525
u_0 = 0
u_1 = 1

f_a0 = f_left(x_0)
f_b0 = f_right(x_0)
k_a0 = k_left(x_0)
k_b0 = k_right(x_0)
q_0 = q(x_0)

lambda_a = sqrt(q_0 / k_a0)
lambda_b = sqrt(q_0 / k_b0)
mu_a = f_a0 / q_0
mu_b = f_b0 / q_0

A_11 = exp(-lambda_a * x_0) - exp(lambda_a * x_0)
A_12 = exp(-lambda_b * (2 - x_0)) - exp(lambda_b * x_0)
A_21 = k_a0 * lambda_a * (exp(-lambda_a * x_0) + exp(lambda_a * x_0))
A_22 = k_b0 * lambda_b * (exp(-lambda_b * (2 - x_0)) - exp(lambda_b * x_0))
B_1 = mu_b - mu_a + (mu_a - u_0) * exp(lambda_a * x_0) - (mu_b - u_1) * exp(lambda_b * (1 - x_0))
B_2 = k_a0 * lambda_a * (-mu_a + u_0) * exp(lambda_a * x_0) + k_b0 * lambda_b * (-mu_b + u_1) * exp(
    lambda_b * (1 - x_0))

c_1 = (((u_0 - mu_a) * A_11 - B_1) * A_22 - ((u_0 - mu_a) * A_21 - B_2) * A_12) / (A_11 * A_22 - A_12 * A_21)
c_2 = (B_1 * A_22 - B_2 * A_12) / (A_11 * A_22 - A_12 * A_21)
c_3 = (B_2 * A_11 - B_1 * A_21) / (A_11 * A_22 - A_12 * A_21)
c_4 = (u_1 - mu_b) * exp(lambda_b) - c_3 * exp(2 * lambda_b)


def u_left(x):
    return c_1 * exp(lambda_a * x) + c_2 * exp(-lambda_a * x) + mu_a


def u_right(x):
    return c_3 * exp(lambda_b * x) + c_4 * exp(-lambda_b * x) + mu_b


def k_a(x, minus=False, h=0.1):
    return k_left(x + (h / 2) if not minus else -(h / 2))


def k_b(x, minus=False, h=0.1):
    return k_right(x + (h / 2) if not minus else -(h / 2))


def solve_left():
    pass


L = 10


def solve_model(h, l_alpha, l_beta):
    a_a = k_a0
    b_a = -2 * k_a0 - q_0 * h * h
    c_a = k_a0
    d_a = -f_a0 * h * h
    a_b = k_b0
    b_b = -2 * k_b0 - q_0 * h * h
    c_b = k_b0
    d_b = -f_b0 * h * h

    alpha_left = [-a_a / b_a, ]
    beta_left = [(-d_a - c_a * u_0) / b_a, ]

    for i in range(1, l_alpha - 1):
        alpha_left.append(-a_a / (b_a + c_a * alpha_left[i - 1]))
        beta_left.append((d_a - c_a * beta_left[i - 1]) / (b_a + a_a * alpha_left[i - 1]))

    print('length of alpha_left is', len(alpha_left))

    alpha_right = [-c_b / b_b]
    beta_right = [(d_b - c_b * u_1) / b_b]

    for i in range(L - 2, l_beta, -1):
        beta_right.insert(0, ((d_b - a_a * beta_right[0]) / (b_b + a_b * alpha_right[0])))
        alpha_right.insert(0, -c_b / (b_b + a_b * alpha_right[0]))

    print('length of alpha_right is', len(alpha_right))

    u = 2 * [
        (k_a0 * beta_left[-1] + k_b0 * beta_right[0]) / (k_a0 * (1 - alpha_left[-1]) + k_b0 * (1 - alpha_right[0])), ]

    for alpha, beta in zip(alpha_right, beta_right):
        u.append(alpha * u[-1] + beta)

    for alpha, beta in reversed(list(zip(alpha_left, beta_left))):
        u.insert(0, alpha * u[0] + beta)

    u.insert(u_0, 0)
    u.append(u_1)
    return u


l_a = int(L / 2)
precise_model = [u_left(x) if i <= l_a else u_right(x) for i, x in enumerate(grid)]
numeric_model = solve_model(0.1, l_a, l_a + 1)
print(precise_model, numeric_model, sep='\n')
print(len(precise_model), len(numeric_model), sep='\n')
