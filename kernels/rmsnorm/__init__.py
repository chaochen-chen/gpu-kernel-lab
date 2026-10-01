"""RMSNorm implementations. Every impl has the signature f(x: [M, N], weight: [N]) -> [M, N]."""

from .reference import rmsnorm_ref
from .triton_rmsnorm import rmsnorm_triton

REF = rmsnorm_ref

IMPLS = {
    "triton": rmsnorm_triton,
    # "liger": ...  # add once you wire up liger_kernel.ops.rms_norm for comparison
}
