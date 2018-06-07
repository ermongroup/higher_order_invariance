import numpy as np
from scipy.special import gamma, polygamma, digamma

init_alpha = 1
init_beta = 1


def gen_samples(alpha, beta, num=100):
    data = np.random.gamma(shape=alpha, scale=1/beta, size=num)
    global s_x, s_logx
    s_x = np.sum(data) / num
    s_logx = np.sum(np.log(data)) / num
    return data

def nll(alpha, beta):
    return -alpha * np.log(beta) + np.log(gamma(alpha)) - (alpha -  1) * s_logx + beta * s_x

def metric(alpha, beta):
    return np.array([
        [polygamma(1, alpha), -1/beta],
        [-1/beta, alpha / beta**2]
    ])

def inverse_metric(alpha, beta):
    return np.array([
        [alpha, beta],
        [beta, beta**2 * polygamma(1, alpha)]
    ]) / (alpha * polygamma(1, alpha) - 1.)

def dL(alpha, beta):
    return np.array(
        [-np.log(beta) + digamma(alpha) - s_logx,
         s_x - alpha / beta]
    )

def Cxx(alpha, beta, params):
    C111 = 1/2 * alpha * polygamma(2, alpha)
    C112 = 1/2 / beta
    C122 = -alpha / (2 * beta**2)
    C211 = 1/2 * beta * polygamma(2, alpha)
    C212 = 1/2 * polygamma(1, alpha)
    C222 = (1/(2 * beta) - alpha * polygamma(1, alpha) / beta)

    dim1 = C111 * params[0] ** 2 + C112 * params[0] * params[1] * 2 + C122 * params[1] ** 2
    dim2 = C211 * params[0] ** 2 + C212 * params[0] * params[1] * 2 + C222 * params[1] ** 2
    return np.array([dim1, dim2]) / (alpha * polygamma(1, alpha) - 1.)
