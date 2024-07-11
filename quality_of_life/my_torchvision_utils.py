import torch
import torchvision

def hot_1_encode_an_integer( n_class, dtype=torch.int64 ):
    return torchvision.transforms.Lambda(
            #
            # ~~~ function that sends an integer y between 0 and n_class-1 to the (y+1)-th standard basis vector in R^{n_class} 
            lambda y: torch.zeros( n_class, dtype=dtype ).scatter_( dim=0, index=torch.tensor(y), value=1 )
        )
