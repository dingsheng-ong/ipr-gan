import numpy as np
import random
import torch
import time


def set_seed(seed=None):
    """
    Set random seed
    """
    if not seed: seed = int(time.time())

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    """
    Test set SEED for RNG
    """
    print('========== SET SEED ==========')
    start = time.time()
    set_seed()
    end   = time.time()
    print(f'Time taken: {(end - start) * 1e3:.2f} ms')
