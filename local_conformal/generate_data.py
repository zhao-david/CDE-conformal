import numpy as np
from scipy.stats import truncnorm


def generate_data_dz(size=1000):
    """
    X ~ Norm(0,1)
    Y|X=x ~ 0.5 N(x,1)+0.5 N(x,0.01)
    """
    X = np.random.normal(0, 1, size)
    ind_mix = np.random.binomial(n=1, p=0.5, size=size)
    mix1 = np.random.normal(X, 1)
    mix2 = np.random.normal(X, 0.1)
    Y = ind_mix * mix1 + (1-ind_mix) * mix2
    return X, Y


def generate_data_b1(size=1000):
    """
    X ~ Unif(-2,2)
    Y|X=x ~ 0.5 N(x,1)+0.5 N(x,0.01)
    """
    X = np.random.uniform(-2, 2, size)
    ind_mix = np.random.binomial(n=1, p=0.5, size=size)
    mix1 = np.random.normal(X, 1)
    mix2 = np.random.normal(X, 0.1)
    Y = ind_mix * mix1 + (1-ind_mix) * mix2
    return X, Y

def _generate_data_b1_cond_inner(X, size = 300):
    """
    Y|X=x ~ 0.5 N(x,1)+0.5 N(x,0.01)

    x is a singleton
    """
    X_all = np.repeat(X, size)
    ind_mix = np.random.binomial(n=1, p=0.5, size=size)
    mix1 = np.random.normal(X, 1)
    mix2 = np.random.normal(X, 0.1)
    Y = ind_mix * mix1 + (1-ind_mix) * mix2

    return Y

def generate_data_b1_cond(X, size = 300):
    """
    Y|X=x ~ 0.5 N(x,1)+0.5 N(x,0.01)
    """

    out = [_generate_data_b1_cond_inner(x, size = size) for x in X]

    return (out)



def generate_data_b2(size=1000):
    """
    X ~ Unif(-2,2)
    Y|X=x ~ 0.5 N(x,1)+0.5 N(x + .25*|x|,0.01)
    """
    X = np.random.uniform(-2, 2, size)
    ind_mix = np.random.binomial(n=1, p=0.5, size=size)
    mix1 = np.random.normal(X, 1)
    mix2 = np.random.normal(X + .25*np.abs(X), 0.1)
    Y = ind_mix * mix1 + (1-ind_mix) * mix2
    return X, Y


def _generate_data_b2_cond_inner(X, size = 300):
    """
    Y|X=x ~ 0.5 N(x,1)+0.5 N(x,0.01)

    x is a singleton
    """
    X_all = np.repeat(X, size)
    ind_mix = np.random.binomial(n=1, p=0.5, size=size)
    mix1 = np.random.normal(X, 1)
    mix2 = np.random.normal(X + .25*np.abs(X), 0.1)
    Y = ind_mix * mix1 + (1-ind_mix) * mix2

    return Y

def generate_data_b2_cond(X, size = 300):
    """
    Y|X=x ~ 0.5 N(x,1)+0.5 N(x,0.01)
    """

    out = [_generate_data_b2_cond_inner(x, size = size) for x in X]

    return (out)


def generate_data_b3(size=1000):
    """
    X ~ Unif(-2,2)
    Y|X=x ~ 0.5 N(x,1)+0.5 N(x + .25*|x|,0.01)
    """
    X = np.random.uniform(-2, 2, size)
    ind_mix = np.random.binomial(n=1, p=0.5, size=size)
    mix1 = np.random.normal(X, 1)
    mix2 = np.random.normal(X + .25*np.abs(X), 0.1)
    Y = ind_mix * mix1 + (1-ind_mix) * mix2
    return X, Y



def generate_data_b0(size = 1000):
    """
    X : uniform(-4,4)

    X'|X = x:
        if (x >= 0):
            x - 2
        else:
            -1 * (x + 2)


    Y |X' = x':
    sd(x') =  1 + 1.5 * |x'|
    truncated_max(x') = .5 + log(2/|x'|)
    df(x') =  1/8*(2*(3 - |x'|))^3 + 2

    \bar{y} = x'

    if x' <= 0:
        y = \bar{y} + truncated_norm_rvs(sd = sd(x),
                                     lower = -1*truncated_max(x)*sd(x),
                                     max = truncated_max(x)*sd(x))
    else:
        y = \bar{y} + sd(x') * student_t(df(x'))
    """

    x = np.random.uniform(-4,4, size)
    x2 = x.copy()
    x2[x >= 0] = x[x >= 0] - 2
    x2[x < 0] = -1*(x[x < 0] + 2)

    sd_val = 1 + 1.5 * np.abs(x2)
    bb = 2/3*(.5 + np.log(2/np.abs(x2))) * sd_val
    aa = -1 * bb

    df = 1/8 * (2*(3- np.abs(x2)))**3 + 2
    y = x.copy()


    y[x2 <= 0] += truncnorm.rvs(a = aa[x2 <= 0],
                               b = bb[x2 <= 0],
                               loc = np.zeros(np.sum(x2 <= 0)),
                               scale = sd_val[x2 <= 0],
                               size = np.sum(x2 <= 0))
    y[x2 > 0] += sd_val[x2 > 0] * np.random.standard_t(df[x2 > 0],
                                                     size = np.sum(x2 > 0))


    return x,y


def _generate_data_b0_cond_inner(X, size = 300):
    """
    X'|X = x:
        if (x >= 0):
            x - 2
        else:
            -1 * (x + 2)


    Y |X' = x':
    sd(x') =  1 + 1.5 * |x'|
    truncated_max(x') = .5 + log(2/|x'|)
    df(x') =  1/8*(2*(3 - |x'|))^3 + 2

    \bar{y} = x'

    if x' <= 0:
        y = \bar{y} + truncated_norm_rvs(sd = sd(x),
                                     lower = -1*truncated_max(x)*sd(x),
                                     max = truncated_max(x)*sd(x))
    else:
        y = \bar{y} + sd(x') * student_t(df(x'))

    x is a singleton
    """
    X_all = np.repeat(X, size)
    x2 = X_all.copy()
    x2[x >= 0] = x[x >= 0] - 2
    x2[x < 0] = -1*(x[x < 0] + 2)

    sd_val = 1 + 1.5 * np.abs(x2)
    bb = 2/3*(.5 + np.log(2/np.abs(x2))) * sd_val
    aa = -1 * bb

    df = 1/8 * (2*(3- np.abs(x2)))**3 + 2
    y = x.copy()


    y[x2 <= 0] += truncnorm.rvs(a = aa[x2 <= 0],
                               b = bb[x2 <= 0],
                               loc = np.zeros(np.sum(x2 <= 0)),
                               scale = sd_val[x2 <= 0],
                               size = np.sum(x2 <= 0))
    y[x2 > 0] += sd_val[x2 > 0] * np.random.standard_t(df[x2 > 0],
                                                     size = np.sum(x2 > 0))



    return y

def generate_data_b0_cond(X, size = 300):
    """
    X'|X = x:
        if (x >= 0):
            x - 2
        else:
            -1 * (x + 2)


    Y |X' = x':
    sd(x') =  1 + 1.5 * |x'|
    truncated_max(x') = .5 + log(2/|x'|)
    df(x') =  1/8*(2*(3 - |x'|))^3 + 2

    \bar{y} = x'

    if x' <= 0:
        y = \bar{y} + truncated_norm_rvs(sd = sd(x),
                                     lower = -1*truncated_max(x)*sd(x),
                                     max = truncated_max(x)*sd(x))
    else:
        y = \bar{y} + sd(x') * student_t(df(x'))
    """

    out = [_generate_data_b0_cond_inner(x, size = size) for x in X]

    return (out)
