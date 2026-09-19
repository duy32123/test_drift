"""Run on the TARGET machine; CPU tests do not certify CUDA/Blackwell support."""
import argparse
import json
import platform
import time
import torch
from drift_cil.optim import ns5,SpectralProposal,MatrixSteps,Transaction

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--require-cuda",action="store_true")
    args=parser.parse_args()
    out={"python":platform.python_version(),"torch":torch.__version__,
         "cuda_build":torch.version.cuda,"cuda_available":torch.cuda.is_available()}
    if args.require_cuda and not torch.cuda.is_available():
        print(json.dumps(out,indent=2));raise SystemExit("CUDA unavailable")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":
        out.update(gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
                   compiled_architectures=torch.cuda.get_arch_list(),bf16=torch.cuda.is_bf16_supported())
    torch.manual_seed(7)
    for shape in ((16,768),(768,16),(16,3072),(3072,16)):
        p=torch.nn.Parameter(torch.randn(*shape,device=device)*.02)
        (p@torch.randn(shape[1],4,device=device)).square().mean().backward()
        opt=MatrixSteps([("p",p)])
        start=time.perf_counter()
        direction=opt.directions(1e-4)
        basis=SpectralProposal(opt.params,direction)
        proposal=basis.directions(__import__('numpy').ones(basis.size))
        before=p.detach().clone()
        tx=Transaction(opt.params,proposal);tx.apply();tx.restore()
        assert torch.equal(p,before)
        assert torch.isfinite(proposal["p"]).all()
        if device.type=="cuda":torch.cuda.synchronize()
        out[str(shape)]={"matmul_backward_svd_ns5_restore":"passed",
                         "elapsed_seconds":time.perf_counter()-start}
    print(json.dumps(out,indent=2))

if __name__=="__main__":main()
