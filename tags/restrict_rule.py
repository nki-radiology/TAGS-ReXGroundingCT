import os, re, numpy as np
from scipy import ndimage
from build_ensemble_submission import side_region
from eval_t1_val200 import named_region
_LOBEW = re.compile(r"\blobe\b", re.I)
def restrict_mask(ts, text, side_k=12, lobe_k=None):
    """LOBE_K rule (text names a lobe -> named lobe dilated lobe_k) else side dilated side_k; None if no side."""
    lobe_k = int(os.environ.get("LOBE_K", "0")) if lobe_k is None else lobe_k
    st = ndimage.generate_binary_structure(2, 1)
    if lobe_k > 0 and _LOBEW.search(text):
        lob = named_region(text)
        if lob:
            m = np.isin(ts, lob); return np.stack([ndimage.binary_dilation(m[:, :, z], st, iterations=lobe_k) for z in range(m.shape[2])], axis=2)
    sid = side_region(text)
    if sid is None: return None
    m = np.isin(ts, sid); return np.stack([ndimage.binary_dilation(m[:, :, z], st, iterations=side_k) for z in range(m.shape[2])], axis=2)
