
# ~~~ Tom Winckelman wrote this; maintained at https://github.com/ThomasLastName/quality_of_life

import sys
import math
import torch
from tqdm import tqdm
from quality_of_life.my_base_utils import support_for_progress_bars, my_warn

#
# ~~~ Set the random seed for pytorch, numpy, and the base random module all at once
def torch_seed(semilla):    
    torch.manual_seed(semilla)
    if "numpy" in sys.modules.keys():
        sys.modules["numpy"].random.seed(semilla)
    if "random" in sys.modules.keys():
        sys.modules["random"].seed(semilla)

#
# ~~~ Generate 1d training and test data, where lables y are generated by y=f(x)
def generate_random_1d_data( ground_truth, n_train, n_test=1001, a=-1, b=1, noise=0., require_endpoints=True, device="cuda" if torch.cuda.is_available() else "cpu" ):
    #
    # ~~~ First, generate the training data
    x_train = torch.rand( n_train-2 if require_endpoints else n_train )*(b-a)+a
    if require_endpoints:   # in this case, sample 2 fewer points, ane add the two endpoints
        x_train = torch.cat(( torch.tensor([a]), x_train, torch.tensor([b]) ))
    x_train = x_train.to(device)
    assert len(x_train) == n_train
    #
    #~~~ Obtain labels by applying the "ground truth" f to x_train, and perhaps also corrupting with noise
    y_train = ground_truth(x_train) + noise*torch.randn( size=x_train.shape, device=device )
    #
    #~~~ Finally, generate the (clean and plentiful) test data
    x_test = torch.linspace( a, b ,n_test, device=device )
    y_test = ground_truth(x_test)
    return x_train, y_train, x_test, y_test

#
# ~~~ Extract the raw tensors from a pytorch Dataset
def convert_Dataset_to_Tensors( object_of_class_Dataset, batch_size=None ):
    assert isinstance( object_of_class_Dataset, torch.utils.data.Dataset )
    if isinstance( object_of_class_Dataset, convert_Tensors_to_Dataset ):
        return object_of_class_Dataset.X, object_of_class_Dataset.y
    else:
        n_data = len(object_of_class_Dataset)
        b = n_data if batch_size is None else batch_size
        return next(iter(torch.utils.data.DataLoader( object_of_class_Dataset, batch_size=b )))  # return the actual tuple (X,y)

#
# ~~~ Convert Tensors into a pytorch Dataset; from https://fmorenovr.medium.com/how-to-load-a-custom-dataset-in-pytorch-create-a-customdataloader-in-pytorch-8d3d63510c21
class convert_Tensors_to_Dataset(torch.utils.data.Dataset):
    #
    # ~~~ Define attributes
    def __init__( self, X_tensor, y_tensor, X_transforms_list=None, y_transforms_list=None, **kwargs ):
        super().__init__( **kwargs )
        assert isinstance(X_tensor,torch.Tensor)
        assert isinstance(y_tensor,torch.Tensor)
        assert X_tensor.shape[0]==y_tensor.shape[0]
        self.X = X_tensor
        self.y = y_tensor
        self.X_transforms = X_transforms_list
        self.y_transforms = y_transforms_list
    #
    # ~~~ Method which pytorch requres custom Dataset subclasses to have to enable indexing; see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    def __getitem__(self, index):
        x = self.X[index]
        if self.X_transforms is not None:
            for transform in self.X_transforms: 
                x = transform(x)
        y = self.y[index]
        if self.y_transforms is not None:
            for transform in self.y_transforms: 
                y = transform(y)
        return x, y
    #
    # ~~~ Method which pytorch requres custom Dataset subclasses to have; see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    def __len__(self):
        return self.y.shape[0]

#
# ~~~ Routine that defines the transform that sends an integer n to the (n+1)-th standard basis vector of R^{n_class} (for n in the range 0 <= n < n_class)
def hot_1_encode_an_integer( n_class, dtype=None ):
    def transform(y,dtype=dtype):
        y = y if isinstance(y,torch.Tensor) else torch.tensor(y)
        y = y.view(-1,1) if y.ndim==1 else y
        dtype = y.dtype if (dtype is None) else dtype
        return torch.zeros( size=(y.numel(),n_class), dtype=dtype ).scatter_( dim=1, index=y.view(-1,1), value=1 ).squeeze()       
    return transform

#
# ~~~ From a sequential model, extract more or list the list for which model==nn.sequential(list)
def save_sequential_architecture(model):
    return [(type(m), m) for m in model]

#
# ~~~ From the output list of save_sequential_architecture, do essentially model=nn.sequential(list)
def load_sequential_architecture(architecture):
    layers = []
    for layer_type, layer in architecture:
        if layer_type == torch.nn.Linear:
            #
            # ~~~ For linear layers, create a brand new linear layer of the same size independent of the original
            layers.append(torch.nn.Linear(layer.in_features,layer.out_features))
        else:
            #
            # ~~~ Foor other layers (activations, Flatten, softmax, etc.) just copy it
            layers.append(layer)
    return torch.nn.Sequential(*layers)


# todo for DataLoading onto GPU: https://stackoverflow.com/questions/65932328/pytorch-while-loading-batched-data-using-dataloader-how-to-transfer-the-data-t

"""
        if verbose<=0:
            #
            # ~~~ Do the training logic, and record any metrics, but print nothing
            for _ in range(epochs):
                for data in training_batches:
                    history["loss"].append(train_step( model, data, loss_fn, optimizer, device ))
                    for key in train_keys:
                        history[key].append(training_metrics[key](model,data))
                    for key in test_keys:
                        history[key].append(test_metrics[key](model,test_data))
                for key in epoch_keys:
                    history[key].append(epochal_metrics[key](model,training_batches,test_data))
        elif verbose<=1:
            #
            # ~~~ An individual long-ass progress bar for all epochs combined
            with alive_bar( epochs*len(training_batches), bar="classic" ) as bar:
                for e in range(epochs):
                    for data in training_batches:
                        loss = train_step( model, data, loss_fn, optimizer, next(model.parameters()).device ) 
                        bar.text(f"loss : {loss:.4}")
                        history["loss"].append(loss)
                        for key in train_keys:
                            history[key].append(training_metrics[key](model,data))
                        for key in test_keys:
                            history[key].append(test_metrics[key](model,test_data))
                        bar()
                    for key in epoch_keys:
                        history[key].append(epochal_metrics[key](model,training_batches,test_data))                        
        else:
            #
            # ~~~ keras-style verbose==2 progress bars: a new progress bar for every epoch
            for e in range(epochs):
                with alive_bar( len(training_batches), bar="classic", title=f"Epoch {e+1}" ) as bar:
                    for data in training_batches:
                        loss = train_step( model, data, loss_fn, optimizer, next(model.parameters()).device )
                        history["loss"].append(loss)
                        msg = f"loss {loss:.4}"
                        for key in train_keys:
                            value = training_metrics[key](model,data)
                            history[key].append(value)
                            msg += f" | {key}: {value:.3} "
                        for key in test_keys:
                            value = test_metrics[key](model,test_data)
                            history[key].append(value)
                            msg += f" | {key}: {value:.3} "
                        bar.text(msg.strip(" "))
                        bar()
                msg = ""
                for key in epoch_keys:
                    val = epochal_metrics[key](model,training_batches,test_data)
                    msg+= f" | {key}: {val:.3} "
                    history[key].append(val)
                bar.text(msg.strip(" "))
    return history
    


class Model(torch.nn.Sequential):
    def __init__( self, *args, **kwargs ):
        super().__init__(*args,**kwargs)
        self.loss_fn = None
        self.optimizer = None
        self.train_step = standard_train_step
    def __getattr__(self,name):
        if name == "loss_fn":
            return self.loss_fn
        elif name == "optimizer":
            return self.optimizer
        elif name == "train_step":
            return self.train_step
        else:
            # Delegate attribute access to the parent class
            return getattr(super(),name)
    def fit( self, training_batches, test_data=None, epochs=20, verbose=2, epochal_metrics=None, training_metrics=None, test_metrics=None ):
        train_keys = training_metrics.keys() if isinstance(training_metrics,dict) else None
        test_keys  = test_metrics.keys()     if isinstance(test_metrics,dict) else None
        epoch_keys = epochal_metrics.keys()  if isinstance(epochal_metrics,dict) else None
        combined_keys = set({"loss"})
        if train_keys is not None:
            combined_keys = combined_keys | set(train_keys)
        if test_keys is not None:
            combined_keys = combined_keys | set(test_keys)
        if epoch_keys is not None:
            combined_keys = combined_keys | set(epoch_keys) 
        history = {key: [] for key in combined_keys}
        #
        # ~~~ The actual training logic
        with support_for_progress_bars():
            if verbose<=0:
                #
                # ~~~ Do the training logic, and record any metrics, but print nothing
                for _ in range(epochs):
                    for data in training_batches:
                        history["loss"].append(self.train_step(self,data).item())
                        for key in train_keys:
                            history[key].append(training_metrics[key](self,data))
                        for key in test_keys:
                            history[key].append(test_metrics[key](self,test_data))
                    for key in epoch_keys:
                        history[key].append(epochal_metrics[key](self,training_batches,test_data))
            elif verbose<=1:
                #
                # ~~~ An individual long-ass progress bar for all epochs combined
                with alive_bar( epochs*len(training_batches), bar="classic" ) as bar:
                    for e in range(epochs):
                        for data in training_batches:
                            loss = self.train_step(self,data).item()
                            bar.text(f"loss : {loss:.4}")
                            history["loss"].append(loss)
                            for key in train_keys:
                                history[key].append(training_metrics[key](self,data))
                            for key in test_keys:
                                history[key].append(test_metrics[key](self,test_data))
                            bar()
                        for key in epoch_keys:
                            history[key].append(epochal_metrics[key](self,training_batches,test_data))                        
            else:
                #
                # ~~~ keras-style verbose==2 progress bars: a new progress bar for every epoch
                for e in range(epochs):
                    with alive_bar( len(training_batches), bar="classic", title=f"Epoch {e}" ) as bar:
                        for data in training_batches:
                            loss = self.train_step(self,data).item()
                            history["loss"].append(loss)
                            msg = f"loss {loss:.4}"
                            for key in train_keys:
                                msg += f" | {key}: {training_metrics[key](self,data):.3} "
                            for key in test_keys:
                                msg += f" | {key}: {test_metrics[key](self,test_data):.3} "
                            bar.text(msg.strip(" "))
                            bar()
                    msg = ""
                    for key in epoch_keys:
                        val = epochal_metrics[key](self,training_batches,test_data)
                        msg+= f" | {key}: {val:.3} "
                        history[key].append(val)
                    bar.text(msg.strip(" "))
"""

#
# ~~~ A normal Johnson–Lindenstrauss matrix, which projects to a lower dimension while approximately preserving pairwise distances
def JL_layer( in_features, out_features ):
    linear_embedding = torch.nn.Linear( in_features, out_features, bias=False )
    linear_embedding.weight.requires_grad = False
    torch.nn.init.normal_( linear_embedding.weight, mean=0, std=1/math.sqrt(out_features) )
    return linear_embedding

#
# ~~~ Apply JL "offline" to a dataset
def embed_dataset( dataset, embedding, device=("cuda" if torch.cuda.is_available() else "cpu") ):
    # embedding = JL_layer(D,d).to(device) if embedding is None else embedding
    batches_of_data = torch.utils.data.DataLoader( dataset, batch_size=100, shuffle=False )
    with support_for_progress_bars():
        for j, (X,y) in enumerate(tqdm(batches_of_data)):
            X = X.to(device)
            y = y.to(device)
            if j==0:
                embedded_X = embedding(X)
                the_same_y = y
            else:
                embedded_X = torch.row_stack(( embedded_X, embedding(X) ))
                the_same_y = torch.row_stack(( the_same_y, y ))
    return embedded_X, the_same_y


# TODO: faster is to simply generate `embedding=torch.randn(d,D,device="cuda")>0`
#       if you just torch.save(embedding,"boolean.pt") it will be memory efficient, as will torch.load("boolean.pt")
#       to convert to float, you need to three separate lines:
#           embedding = embedding.float()
#           embedding *= 2
#           embedding -= 1
#       in contrast, embedding=2*embedding.float()-1 gives a memory error
#       on the other hand, e_float@x == ( (x*e_bool) - (x*~e_bool) ).sum(axis=1) when e_bool = torch.randn(d,D,device="cuda")>0 and e_float = 2*e_bool-1.
#       e_float@x is perhaps faster, but the latter is more memory efficient, as e_float never needs to be loaded in memory



class SkipNet(torch.nn.Module):
    def __init__(self, *layers):
        super(SkipNet, self).__init__()
        self.layers = torch.nn.ModuleList(layers)
    def forward(self,x):
        #
        # ~~~ Make a copy of x in the form that it is originally supplied
        original_input = x.clone()
        #
        # ~~~ Apply each layer and, after each non-linear, non-final, add a skip connection
        for j,layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer,torch.nn.Linear) or (j+1)==len(self.layers):
                pass                # ~~~ do *not* add a skip connection after a linear layer, nor after the final layer
            else:
                x += original_input # ~~~ add a skip connection after each non-linear layer, except never after the last layer
        #
        # ~~~ Done
        return x

# # Test the model
# x = torch.randn(1, 1)  # Example input tensor
# model = SkipNet(
#     torch.nn.Linear(1, 100),
#     torch.nn.ReLU(),
#     torch.nn.Linear(100, 1)
# )
# print(model(x))

#
# ~~~ A shallow, univariate residual network
class TinyResNet(torch.nn.Module):
    def __init__( self, width, activation=torch.nn.ReLU() ):
        super(TinyResNet,self).__init__()
        self.inner_dense = torch.nn.Linear(1,width) # ~~~ fully connected layer
        self.activation = activation                # ~~~ activation function
        self.outer_dense = torch.nn.Linear(width,1) # ~~~ fully connected layer
    def forward(self, x):
        return self.outer_dense(self.activation(self.inner_dense(x))) + x

#
# ~~~ Get all available gradients of the parameters in a pytorch model
def get_flat_grads(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    return torch.cat(grads)

#
# ~~~ Given a flat vector of desired gradients, and a model, assign those to the .grad attribute of the model's parameters
def set_flat_grads(model,flat_grads):
    # TODO: a safety feature checking the shape/class of flat_grads (should be a 1d Torch vector)
    start = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.grad.numel()
            p.grad.data = flat_grads[start:start+numel].view_as(p.grad)
            start += numel
    if start>len(flat_grads):
        my_warn(f"The lenght of the supplied vector [{len(flat_grads)}] exceeds the number of parameters in the model which require grad [{start}]")

#
# ~~~ Helper function which creates a new instance of the supplied sequential architeture
def nonredundant_copy_of_module_list(module_list,sequential=False):
    architecture = [ (type(layer),layer) for layer in module_list ]
    layers = []
    for layer_type, layer in architecture:
        if layer_type == torch.nn.Linear:
            #
            # ~~~ For linear layers, create a brand new linear layer of the same size independent of the original
            layers.append(torch.nn.Linear( layer.in_features, layer.out_features ))
        else:
            #
            # ~~~ For other layers (activations, Flatten, softmax, etc.) just copy it
            layers.append(layer)
    return torch.nn.Sequential(*layers) if sequential else torch.nn.ModuleList(layers)

#
# ~~~ Return the len(x)-by-len(y) matrix Z matrix with Z[i,j] = f([x[i],y[j]])
def apply_on_cartesian_product(f,x,y):
    X,Y = torch.meshgrid( x, y, indexing="xy" )
    cartesian_product = torch.column_stack((X.flatten(), Y.flatten())) # ~~~ the result is basically just a rearranged version of list(itertools.product(x,y))
    return f(cartesian_product).reshape(X.shape)