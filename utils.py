import numpy as np

def normalize_vec(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    else:
        return vec / norm