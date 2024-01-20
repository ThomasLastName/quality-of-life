
import sys
import torch

def torch_seed(semilla):    
    torch.manual_seed(semilla)
    if "numpy" in sys.modules.keys():
        sys.modules["numpy"].random.seed(semilla)
    if "random" in sys.modules.keys():
        sys.modules["random"].seed(semilla)

