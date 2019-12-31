#!/usr/bin/env python
import random

import numpy as np
import torch

from onmt.bin.rl_train import main


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2019)

if __name__ == "__main__":
    main()
