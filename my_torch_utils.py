
import sys
from types import NoneType
import torch
from alive_progress import alive_bar
from quality_of_life.my_base_utils import support_for_progress_bars, my_warn

def torch_seed(semilla):    
    torch.manual_seed(semilla)
    if "numpy" in sys.modules.keys():
        sys.modules["numpy"].random.seed(semilla)
    if "random" in sys.modules.keys():
        sys.modules["random"].seed(semilla)


def convert_Dataset_to_Tensors( object_of_class_Dataset ):
    assert isinstance( object_of_class_Dataset, torch.utils.data.Dataset )
    n_data=len(object_of_class_Dataset)
    return next(iter(torch.utils.data.DataLoader( object_of_class_Dataset, batch_size=n_data )))  # return the actual tuple (X,y)


#
# ~~~ from https://fmorenovr.medium.com/how-to-load-a-custom-dataset-in-pytorch-create-a-customdataloader-in-pytorch-8d3d63510c21
class convert_Tensors_to_Dataset(torch.utils.data.Dataset):
    #
    # ~~~ Define attributes
    def __init__(self, X_tensor, y_tensor, transform_list=None, **kwargs ):
        super().__init__( **kwargs )
        assert isinstance(X_tensor,torch.Tensor)
        assert isinstance(y_tensor,torch.Tensor)
        assert X_tensor.shape[0]==y_tensor.shape[0]
        self.X = X_tensor
        self.y = y_tensor
        self.transforms = transform_list
    #
    # ~~~ Method which pytorch requres custom Dataset subclasses to have to enable indexing; see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    def __getitem__(self, index):
        x = self.X[index]
        if self.transforms is not None:
            for transform in self.transforms: 
                x = transform(x)
        y = self.y[index]
        return x, y
    #
    # ~~~ Method which pytorch requres custom Dataset subclasses to have; see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    def __len__(self):
        return self.y.shape[0]


def hot_1_encode_an_integer( n_class, dtype=None ):
    def transform(y,dtype=dtype):
        y = y if isinstance(y,torch.Tensor) else torch.tensor(y)
        y = y.view(-1,1) if y.ndim==1 else y
        dtype = y.dtype if (dtype is None) else dtype
        return torch.zeros( size=(y.numel(),n_class), dtype=dtype ).scatter_( dim=1, index=y.view(-1,1), value=1 ).squeeze()       
    return transform


# for DataLoading onto GPU: https://stackoverflow.com/questions/65932328/pytorch-while-loading-batched-data-using-dataloader-how-to-transfer-the-data-t


def standard_train_step( model, data, loss_fn, optimizer, device ):
    #
    # ~~~ Unpack data and move it to the desired device
    X = data[0].to(device)
    y = data[1].to(device)
    #
    # ~~~ Compute prediction error
    loss = loss_fn(model(X),y)
    #
    # ~~~ Backpropagation
    loss.backward()             # ~~~ compute the gradient of loss
    optimizer.step()      # ~~~ apply the gradient to the model parameters
    optimizer.zero_grad() # ~~~ reset the gradient to zero so that we "start fresh" next time standard_train_step is called
    return loss.item()

# todo define a simpler version of fit without all the metrics and ask chat gpt how to incorporate model.evel() and torch.no_grad()
def fit( model, training_batches, loss_fn, optimizer, train_step=standard_train_step, test_data=None, epochs=20, verbose=2, epochal_metrics=None, training_metrics=None, test_metrics=None ):
    #
    # ~~~ Organize keys
    assert isinstance(training_metrics,dict) or training_metrics is None
    assert isinstance(test_metrics,dict) or test_metrics is None
    assert isinstance(epochal_metrics,dict) or epochal_metrics is None
    train_keys = training_metrics.keys() if isinstance(training_metrics,dict) else []
    test_keys  = test_metrics.keys()     if isinstance(test_metrics,dict) else []
    epoch_keys = epochal_metrics.keys()  if isinstance(epochal_metrics,dict) else []
    combined_keys = set({"loss"})
    if not isinstance( train_keys, list ):
        combined_keys = combined_keys | set(train_keys)
    if not isinstance( test_keys, list ):
        combined_keys = combined_keys | set(test_keys)
    if not isinstance( epoch_keys, list ):
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
                    history["loss"].append(train_step( model, data, loss_fn, optimizer, next(model.parameters().device)) )
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
                        bar.text(f"loss : {loss:5.4}")
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
                            bar.text(f"loss : {loss:5.4}")
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
                