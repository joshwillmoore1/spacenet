import numpy as np 
 
def community_ids(label_arr: np.ndarray) -> np.ndarray:
    return np.unique(label_arr[label_arr >= 0])