import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import odeint
import numpy as np
import seaborn as sns
sns.set(font_scale=1.5)
mpl.rcParams['lines.linewidth'] = 3

lr = 0.5
n_iter = 10

def gd():
    theta = np.asarray([init_alpha, init_beta])
    thetas = [theta]
    factor = 1000
    N_iter = factor * n_iter
    small_lr = lr / factor
    nlls = [nll(init_alpha, init_beta)]
    for i in range(N_iter):
        delta_theta = small_lr * dL(theta[0], theta[1])
        theta = theta -  delta_theta
        thetas.append(theta)
        nlls.append(nll(theta[0], theta[1]))

    thetas = [thetas[i] for i in range(0, N_iter + 1, factor)]
    nlls = [nlls[i] for i in range(0, N_iter + 1, factor)]
    nlls = np.asarray(nlls)
    return nlls, thetas

def ng():
    theta = np.asarray([init_alpha, init_beta])
    thetas = [theta]
    nlls = [nll(init_alpha, init_beta)]
    for i in range(n_iter):
        delta_theta = lr * inverse_metric(theta[0], theta[1]) @ dL(theta[0], theta[1])[:, None]
        delta_theta = np.squeeze(delta_theta)
        theta = theta - delta_theta
        thetas.append(theta)
        nlls.append(nll(theta[0], theta[1]))

    print(thetas)
    nlls = np.asarray(nlls)
    return nlls, thetas

def ng_cont():
    theta = np.asarray([init_alpha, init_beta])
    thetas = [theta]
    factor = 1000
    N_iter = factor * n_iter
    small_lr = lr / factor
    nlls = [nll(init_alpha, init_beta)]
    for i in range(N_iter):
        delta_theta = small_lr * inverse_metric(theta[0], theta[1]) @ dL(theta[0], theta[1])[:, None]
        delta_theta = np.squeeze(delta_theta)
        theta = theta -  delta_theta
        thetas.append(theta)
        nlls.append(nll(theta[0], theta[1]))

    thetas = [thetas[i] for i in range(0, N_iter + 1, factor)]
    nlls = [nlls[i] for i in range(0, N_iter + 1, factor)]
    nlls = np.asarray(nlls)
    return nlls, thetas

def geo():
    theta = np.asarray([init_alpha, init_beta])
    thetas = [theta]
    nlls = [nll(init_alpha, init_beta)]
    for i in range(n_iter):
        dot_theta = inverse_metric(theta[0], theta[1]) @ dL(theta[0], theta[1])[:, None]
        dot_theta = np.squeeze(dot_theta)
        theta = theta - lr * dot_theta - 1/2 * lr ** 2 * Cxx(theta[0], theta[1], dot_theta)
        thetas.append(theta)
        nlls.append(nll(theta[0], theta[1]))

    nlls = np.asarray(nlls)
    return nlls, thetas

def geo_cont():
    def f(y, s):
        y0 = y[0]  # u
        y1 = y[1]  # u'
        y2 = y[2]  # v
        y3 = y[3]  # v'
        dy = np.zeros_like(y)
        dy[0] = y1
        dy[2] = y3
        res = -Cxx(y0, y2, [y1, y3])
        dy[1] = res[0]
        dy[3] = res[1]

        return dy

    def exp_map(theta, lr, dot_theta):
        y0 = np.array([theta[0], -lr * dot_theta[0], theta[1], -lr * dot_theta[1]])
        t = np.array([0, 1])
        res, infodict = odeint(f, y0, t, full_output=True)
        res_y = res[-1, :]
        return np.asarray([res_y[0], res_y[2]])

    theta = np.asarray([init_alpha, init_beta])
    thetas = [theta]
    nlls = [nll(init_alpha, init_beta)]
    for _ in range(n_iter):
        dot_theta = inverse_metric(theta[0], theta[1]) @ dL(theta[0], theta[1])[:, None]
        dot_theta = np.squeeze(dot_theta)
        theta = exp_map(theta, lr, dot_theta)
        thetas.append(theta)
        nlls.append(nll(theta[0], theta[1]))

    nlls = np.asarray(nlls)
    return nlls, thetas

def mid():
    theta = np.asarray([init_alpha, init_beta])
    thetas = [theta]
    nlls = [nll(init_alpha, init_beta)]
    for i in range(n_iter):
        delta_theta = lr/2 * inverse_metric(theta[0], theta[1]) @ dL(theta[0], theta[1])[:, None]
        delta_theta = np.squeeze(delta_theta)
        theta_t = theta - delta_theta
        delta_theta = lr * inverse_metric(theta_t[0], theta_t[1]) @ dL(theta_t[0], theta_t[1])[:, None]
        delta_theta = np.squeeze(delta_theta)
        theta = theta - delta_theta
        thetas.append(theta)
        nlls.append(nll(theta[0], theta[1]))

    nlls = np.asarray(nlls)
    return nlls, thetas


def geo_f():
    theta = np.asarray([init_alpha, init_beta])
    thetas = [theta]
    nlls = [nll(init_alpha, init_beta)]
    for i in range(1):
        dot_theta = inverse_metric(theta[0], theta[1]) @ dL(theta[0], theta[1])[:, None]
        dot_theta = np.squeeze(dot_theta)
        theta = theta - lr * dot_theta - 1/2 * lr ** 2 * Cxx(theta[0], theta[1], dot_theta)
        thetas.append(theta)
        nlls.append(nll(theta[0], theta[1]))

    for i in range(n_iter - 1):
        dot_theta = (thetas[-1] - thetas[-2]) / lr
        real_dot_theta = inverse_metric(theta[0], theta[1]) @ dL(theta[0], theta[1])[:, None]
        real_dot_theta = np.squeeze(real_dot_theta)
        theta = theta - lr * real_dot_theta - 1/2 * lr ** 2 * Cxx(theta[0], theta[1], dot_theta)
        thetas.append(theta)
        nlls.append(nll(theta[0], theta[1]))

    nlls = np.asarray(nlls)
    return nlls, thetas

def plot(name, ylabel=False):
    np.random.seed(1234)
    gen_samples(20, 20, 10000)

    nlls_ng, _ = ng()
    nlls_ng_cont, thetas = ng_cont()
    nlls_geo, _ = geo()
    nlls_geo_f, _ = geo_f()
    nlls_mid, _ = mid()
    nlls_geo_cont, _ = geo_cont()

    xrange = np.r_[:len(nlls_ng)]
    plt.plot(xrange, nlls_ng, '-')
    plt.plot(xrange, nlls_mid, '-')
    plt.plot(xrange, nlls_geo, '-')
    plt.plot(xrange, nlls_geo_f, '-')
    plt.plot(xrange, nlls_ng_cont, '--')
    plt.plot(xrange, nlls_geo_cont, '--')
    plt.legend(["ng", "mid", "geo", r"geo$\mathrm{_f}$", "ng(exact)", "geo(exact)"])
    plt.xlabel("# of iterations")
    plt.grid('off')
    if ylabel:
        plt.ylabel("Negative Log-Likelihood")
    plt.title(name)

if __name__ == '__main__':
    from param0 import *
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 4, 1)
    plot(r"$\alpha$, $\beta$", ylabel=True)
    plt.subplot(1, 4, 2)
    from param1 import *
    plot(r"$\alpha \rightarrow \alpha'$, $\beta \rightarrow \frac{1}{\beta'}$")
    from param_beta_3 import *
    plt.subplot(1, 4, 3)
    plot(r"$\alpha \rightarrow \alpha'$, $\beta \rightarrow (\beta')^3$")
    from param_square import *
    plt.subplot(1, 4, 4)
    plot(r"$\alpha \rightarrow (\alpha')^2$, $\beta \rightarrow (\beta')^2$")
    plt.tight_layout()
    plt.savefig('gamma.png',dpi=300, bbox_inches='tight')
    plt.show()
