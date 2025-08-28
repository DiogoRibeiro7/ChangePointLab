# anchor_utils.py
# MIT License
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Dict

import numpy as np
from numpy.typing import NDArray

Tau = Tuple[int, ...]


def rotate_tau(tau: Tau, shift: int, N: int) -> Tau:
    """
    Rotate changepoints by `-shift` on the N-lattice so that positions move left by 'shift'.
    """
    if not tau:
        return ()
    return tuple(sorted(((cp - shift) % N for cp in tau)))


def rotate_array(arr: NDArray, shift: int) -> NDArray:
    """Rotate a 1-D array left by 'shift' (positive shift == left)."""
    shift = int(shift) % arr.size
    if shift == 0:
        return arr.copy()
    return np.concatenate([arr[shift:], arr[:shift]])


def choose_anchor_from_cp_hist(cp_hist: NDArray[np.integer]) -> int:
    """
    Choose an anchor index as the argmax of the changepoint posterior mass.
    """
    return int(np.argmax(cp_hist))


def remap_samples_to_anchor(samples: Sequence[Tau], N: int, anchor: int) -> List[Tau]:
    """
    Rotate each τ so that 'anchor' becomes index 0 in a canonicalized frame.
    This is useful for comparing segment locations relative to a salient reference.
    """
    out: List[Tau] = []
    for tau in samples:
        out.append(rotate_tau(tau, shift=anchor, N=N))
    return out


def rotate_pointwise_summary(pw: Dict[str, NDArray], shift: int) -> Dict[str, NDArray]:
    """
    Rotate a pointwise summary dict {"median","lower","upper"} by 'shift' positions.
    """
    return {k: rotate_array(v, shift) for k, v in pw.items()}



# from anchor_utils import choose_anchor_from_cp_hist, remap_samples_to_anchor, rotate_pointwise_summary
# anchor = choose_anchor_from_cp_hist(result.changepoint_hist)
# samples_rot = remap_samples_to_anchor(result.samples_tau, prior.N, anchor)
# pw_rot = rotate_pointwise_summary(pw, shift=anchor)
