from . import _C

def forward(
    sites,
    colors,
    alpha,
    idx_topk,
    iidx,
    weights,
    wo,
    k_level,
    cell_id,
):
    return _C.forward(
        sites,
        colors,
        alpha,
        idx_topk,
        iidx,
        weights,
        wo,
        k_level,
        cell_id,
    )
