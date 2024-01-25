import torch
import torchvision

def hot_1_encode_an_integer( n_class, dtype=torch.float ):
    return torchvision.transforms.Lambda(
            #
            # ~~~ function that sends an integer y between 0 and n_class-1 to the (y+1)-th standard basis vector in R^{n_class} 
            lambda y: torch.zeros( n_class, dtype=dtype ).scatter_( dim=0, index=torch.tensor(y), value=1 )
        )

train = torchvision.datasets.MNIST(
        root = "C:\\Users\\thoma\\AppData\\Local\\Programs\\Python\\Python310\\pytorch_data",
        train = True,
        download = False,
        transform = torchvision.transforms.ToTensor(),
        target_transform = hot_1_encode_an_integer( n_class=10 )
    )