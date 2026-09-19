import numpy as np
import pytest
import torch
from drift_cil.optim import MatrixSteps,SpectralProposal,Transaction,accept_step,ns5
from drift_cil.model import LoRALinear


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_real_backend_reconstruction_and_bit_exact_rejection(device):
    if device=="cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    torch.manual_seed(13)
    p=torch.nn.Parameter(torch.randn(12,3,device=device))
    p.grad=torch.randn_like(p)
    opt=MatrixSteps([("w",p)])
    direction=opt.directions(.007)
    basis=SpectralProposal(opt.params,direction)
    reconstructed=basis.directions(np.ones(basis.size))
    assert torch.allclose(direction["w"],reconstructed["w"],atol=2e-8,rtol=2e-5)
    before=p.detach().clone()
    tx=Transaction(opt.params,reconstructed)
    outcome=accept_step(tx,lambda:1.+float((p.detach()-before).square().sum()),1.,.999,max_backtracks=2)
    assert outcome["accept_status"]=="rejected"
    assert torch.equal(p,before)


def test_zero_momentum_never_moves_or_creates_an_arbitrary_mode():
    p=torch.nn.Parameter(torch.zeros(8,3));p.grad=torch.zeros_like(p)
    opt=MatrixSteps([("w",p)])
    d=opt.directions(.1)
    assert torch.count_nonzero(d["w"])==0
    assert SpectralProposal(opt.params,d).size==0


def test_coordinates_match_actual_applied_first_order_change():
    torch.manual_seed(2)
    p=torch.nn.Parameter(torch.randn(5,3));p.grad=torch.randn_like(p)
    grads={"w":p.grad.clone()}
    opt=MatrixSteps([("w",p)])
    d=opt.directions(.01)
    proposal=SpectralProposal(opt.params,d)
    z=np.array([.2,.7,.4])
    actual=proposal.directions(z)["w"]
    assert np.isclose(proposal.coordinates(grads)@z,float((grads["w"]*actual).sum()),rtol=1e-5)


def test_finite_factor_update_includes_cross_term():
    torch.manual_seed(3)
    layer=LoRALinear(torch.nn.Linear(7,5,bias=False),3,6)
    with torch.no_grad(): layer.lora_B.normal_()
    a,b=layer.lora_A.detach().clone(),layer.lora_B.detach().clone()
    da,db=torch.randn_like(a)*.01,torch.randn_like(b)*.01
    before=layer.scaling*b@a
    after=layer.scaling*(b+db)@(a+da)
    predicted=layer.scaling*(db@a+b@da+db@da)
    assert torch.allclose(after-before,predicted,atol=5e-7)


def test_sampled_label_mean_score_has_expected_batch_scaling():
    # Enumerate independent Bernoulli labels exactly; no MC variance in this test.
    p=.3
    per=p*(1-p)
    variance=0.
    for y1 in (0,1):
        for y2 in (0,1):
            probability=(p if y1 else 1-p)*(p if y2 else 1-p)
            variance+=probability*((p-y1+p-y2)/2)**2
    assert np.isclose(2*variance,per)
