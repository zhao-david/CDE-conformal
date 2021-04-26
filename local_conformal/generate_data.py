import numpy as np

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



