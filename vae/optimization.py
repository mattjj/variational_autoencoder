import theano
import theano.tensor as T

from util import shared_zeros_like, concat, floatX, argprint


@argprint
def sgd(rate):
    def sgd(cost, params):
        grads = T.grad(cost=cost, wrt=params)
        return [(p, p - g * rate) for p, g in zip(params, grads)]
    return sgd


@argprint
def adagrad(rate, epsilon=1e-6):
    def adagrad(cost, params):
        grads = T.grad(cost=cost, wrt=params)
        sumsq = [shared_zeros_like(p) for p in params]

        def make_update(p, g, c):
            c_new = c + g**2
            p_new = p - rate * g / T.sqrt(c_new + epsilon)
            return [(c, c_new), (p, p_new)]

        return concat(make_update(p,g,c) for p,g,c in zip(params, grads, sumsq))
    return adagrad


@argprint
def rmsprop(rate, rho=0.9, epsilon=1e-6):
    def rmsprop(cost, params):
        grads = T.grad(cost=cost, wrt=params)
        sumsq = [shared_zeros_like(p) for p in params]

        def make_update(p, g, c):
            c_new = rho*c + (1.-rho)*g**2
            p_new = p - rate * g / T.sqrt(c_new + epsilon)
            return [(c, c_new), (p, p_new)]

        return concat(make_update(p, g, c) for p, g, c in zip(params, grads, sumsq))
    return rmsprop


@argprint
def adadelta(rho=0.95, epsilon=1e-6):
    # http://arxiv.org/abs/1212.5701
    def adadelta(cost, params):
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
    return adadelta


@argprint
def adam(alpha, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    # http://arxiv.org/abs/1412.6980
    def adam(cost, params):
        grads = T.grad(cost=cost, wrt=params)
        ms = [shared_zeros_like(p) for p in params]
        vs = [shared_zeros_like(p) for p in params]
        t = theano.shared(floatX(1.))

        def make_update(p, g, m, v, t):
            m_new = beta_1*m + (1.-beta_1)*g
            v_new = beta_2*v + (1.-beta_2)*g**2
            mhat = m / (1.-beta_1**t)
            vhat = v / (1.-beta_2**t)
            p_new = p - alpha * mhat / (T.sqrt(vhat) + epsilon)
            return [(m, m_new), (v, v_new), (p, p_new)]

        return [(t, t+1.)] + concat(
            make_update(p, g, m, v, t) for p,g,m,v in zip(params, grads, ms, vs))
    return adam


@argprint
def momentum_sgd(rate, gamma=0.5):
    # http://jmlr.org/proceedings/papers/v28/sutskever13.pdf
    def momentum_sgd(cost, params):
        grads = T.grad(cost=cost, wrt=params)
        vs = [shared_zeros_like(p) for p in params]

        def make_update(p, g, v):
            v_new = gamma*v - rate*g
            p_new = p + v_new
            return [(v, v_new), (p, p_new)]

        return concat(make_update(p, g, v) for p,g,v in zip(params, grads, vs))
    return momentum_sgd


@argprint
def nesterov(rate, gamma=0.5):
    # http://arxiv.org/abs/1212.0901 Eqs. (6) and (7)
    def nesterov(cost, params):
        grads = T.grad(cost=cost, wrt=params)
        vs = [shared_zeros_like(p) for p in params]

        def make_update(p, g, v):
            v_new = gamma*v - rate*g
            p_new = p + gamma**2*v - (1.+gamma)*rate*g
            return [(v, v_new), (p, p_new)]

        return concat(make_update(p, g, v) for p,g,v in zip(params, grads, vs))
    return nesterov
