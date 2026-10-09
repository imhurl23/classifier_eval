"""Check the custom mixed-logistic likelihood against independent integration."""
import numpy as np
from scipy import integrate,special,stats,optimize
from analysis.inferential_models import objective


def test_glmm_likelihood_matches_independent_adaptive_integration():
    X=np.array([[[1.,0.],[1.,1.],[1.,-.5]],[[1.,0.],[1.,1.],[1.,-.5]]])
    y=np.array([[1.,0.,1.],[0.,1.,0.]])
    theta=np.array([.3,-.7,np.log(.9)])
    expected=0.
    for design,outcomes in zip(X,y):
        def integrand(b):
            eta=design@theta[:-1]+b
            logp=np.sum(outcomes*eta-np.logaddexp(0,eta))
            return np.exp(logp)*stats.norm.pdf(b,scale=np.exp(theta[-1]))
        marginal,error=integrate.quad(integrand,-np.inf,np.inf,epsabs=1e-12,epsrel=1e-12)
        expected-=np.log(marginal)
    assert np.isclose(objective(theta,X,y,121)[0],expected,atol=1e-10)


def test_glmm_reduces_to_bernoulli_likelihood_with_zero_random_variance():
    X=np.array([[[1.,0.],[1.,1.]],[[1.,1.],[1.,-.5]]])
    y=np.array([[1.,0.],[0.,1.]])
    theta=np.array([.3,-.7,-25.])
    eta=np.einsum('gop,p->go',X,theta[:-1])
    expected=-np.sum(y*eta-np.logaddexp(0,eta))
    assert np.isclose(objective(theta,X,y)[0],expected,atol=1e-10)


def test_glmm_analytic_gradient_matches_independent_central_differences():
    rng=np.random.default_rng(123)
    X=rng.normal(size=(8,5,3));X[:,:,0]=1
    y=rng.integers(0,2,size=(8,5)).astype(float)
    theta=np.array([.1,-.3,.7,.2]);h=1e-5
    numeric=[]
    for j in range(len(theta)):
        shift=np.eye(len(theta))[j]*h
        numeric.append((objective(theta+shift,X,y)[0]-objective(theta-shift,X,y)[0])/(2*h))
    assert np.allclose(objective(theta,X,y)[1],numeric,atol=1e-6)
