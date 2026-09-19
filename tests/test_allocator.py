import numpy as np
from scipy.optimize import minimize
from drift_cil.allocator import Curvature, solve_budget


def test_random_feasibility_and_independent_slsqp_objective():
    rng=np.random.default_rng(92)
    for _ in range(25):
        d=6
        scores=rng.normal(size=(12,d))
        c=Curvature(scores,np.full(d,.05))
        a=rng.normal(size=d)*.1
        b=rng.uniform(.1,1.,d)
        rho=.1*c.quad(np.ones(d))
        z,info=solve_budget(b,a,c,rho)
        cost=lambda x: -a@x+.5*c.quad(x)
        other=minimize(lambda x:-b@x,np.zeros(d),jac=lambda x:-b,bounds=[(0,1)]*d,
            constraints=[{"type":"ineq","fun":lambda x:rho-cost(x),"jac":lambda x:a-c.mv(x)}],
            method="SLSQP",options={"maxiter":1000,"ftol":1e-11})
        assert other.success
        assert cost(z)<=rho+1e-7
        assert abs(b@z-b@other.x)<2e-5


def test_recovery_and_unreachable_budget_are_distinguished():
    c=Curvature(np.eye(4),np.full(4,.25))
    a=np.array([1.,0,0,0]);b=np.ones(4)
    z,info=solve_budget(b,a,c,-.2)
    assert info["status"].startswith("recovery")
    assert info["feasible"]
    _,bad=solve_budget(b,-np.ones(4),c,-5)
    assert bad["status"]=="infeasible_candidate"
    assert not bad["feasible"]


def test_negative_coefficients_and_signs():
    # Negative a AND negative b make a negative z help BOTH tasks.
    c=Curvature(np.eye(2),np.zeros(2))
    z,info=solve_budget(-np.ones(2),-np.ones(2),c,.0,lower=-1)
    assert np.all(z<0)
    assert info["objective"]>0 and info["predicted_change"]<0


def test_cross_matrix_covariance_is_not_discarded():
    c=Curvature(np.array([[1.,1.],[-1.,-1.]]),np.zeros(2))
    assert c.quad(np.ones(2))==4
    assert c.quad(np.array([1.,-1.]))==0


def test_learning_rate_changes_curvature_quadratically():
    c=Curvature(np.array([[2.,3.]]),np.array([.1,.1]))
    z=np.ones(2);eta=.001
    scaled=Curvature(c.scores*eta,c.diagonal*eta**2)
    assert np.isclose(scaled.quad(z),eta**2*c.quad(z))
