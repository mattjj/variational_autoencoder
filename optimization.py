import theano
from theano import tensor as T

from util import shared_zeros_like, concat, floatX


def sgd(cost, params, rate):
    grads = T.grad(cost=cost, wrt=params)
    return [(p, p - g * rate) for p, g in zip(params, grads)]


def adagrad(cost, params, rate, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    sumsq = [shared_zeros_like(p) for p in params]

    def make_update(p, g, c):
        c_new = c + g**2
        p_new = p - rate * g / T.sqrt(c_new + epsilon)
        return [(c, c_new), (p, p_new)]

    return concat(make_update(p,g,c) for p,g,c in zip(params, grads, sumsq))


def rmsprop(cost, params, rate, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    sumsq = [shared_zeros_like(p) for p in params]

    def make_update(p, g, c):
        c_new = rho*c + (1.-rho)*g**2
        p_new = p - rate * g / T.sqrt(c_new + epsilon)
        return [(c, c_new), (p, p_new)]

    return concat(make_update(p, g, a) for p, g, a in zip(params, grads, sumsq))


def adadelta(cost, params, rho=0.95, epsilon=1e-6):
    # http://arxiv.org/abs/1212.5701
    grads = T.grad(cost=cost, wrt=params)
    sum_gsq = [shared_zeros_like(p) for p in params]  # accumulated sq. grads
    sum_usq = [shared_zeros_like(p) for p in params]  # accumulated sq. updates

    def make_update(p, g, cg2, cu2):
        cg2_new = rho*cg2 + (1.-rho)*g**2
        ud = -T.sqrt(cu2 + epsilon) / T.sqrt(cg2_new + epsilon) * g
        cu2_new = rho*cu2 + (1.-rho)*ud**2
        p_new = p + ud
        return [(cg2, cg2_new), (cu2, cu2_new), (p, p_new)]

    return concat(make_update(p, g, g2, up2)
                  for p, g, g2, up2 in zip(params, grads, sum_gsq, sum_usq))


def adam(cost, params, alpha, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    # http://arxiv.org/abs/1412.6980
    grads = T.grad(cost=cost, wrt=params)
    ms = [shared_zeros_like(p) for p in params]
    vs = [shared_zeros_like(p) for p in params]
    t = theano.shared(floatX(1))

    def make_update(p, g, m, v, t):
        m_new = beta_1*m + (1.-beta_1)*g
        v_new = beta_2*v + (1.-beta_2)*g**2
        mhat = m / (1.-beta_1**t)
        vhat = v / (1.-beta_2**t)
        p_new = p - alpha * mhat / (T.sqrt(vhat) + epsilon)
        return [(m, m_new), (v, v_new), (p, p_new)]

    return [(t, t+1)] + concat(
        make_update(p, g, m, v, t) for p,g,m,v in zip(params, grads, ms, vs))


def momentum_sgd(cost, params, rate, gamma=0.5):
    # http://jmlr.org/proceedings/papers/v28/sutskever13.pdf
    grads = T.grad(cost=cost, wrt=params)
    vs = [shared_zeros_like(p) for p in params]

    def make_update(p, g, v):
        v_new = gamma*v - rate*g
        p_new = p + v_new
        return [(v, v_new), (p, p_new)]

    return concat(make_update(p, g, v) for p,g,v in zip(params, grads, vs))


def nesterov(cost, params, rate, gamma=0.5):
    # http://arxiv.org/abs/1212.0901 Eqs. (6) and (7)
    grads = T.grad(cost=cost, wrt=params)
    vs = [shared_zeros_like(p) for p in params]

    def make_update(p, g, v):
        v_new = gamma*v - rate*g
        p_new = p + gamma**2*v - (1.+gamma)*rate*g
        return [(v, v_new), (p, p_new)]

    return concat(make_update(p, g, v) for p,g,v in zip(params, grads, vs))
