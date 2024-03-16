
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
# ~~~ Extract the raw tensors from a pytorch Dataset
def convert_Dataset_to_Tensors( object_of_class_Dataset ):
    assert isinstance( object_of_class_Dataset, torch.utils.data.Dataset )
    n_data=len(object_of_class_Dataset)
    return next(iter(torch.utils.data.DataLoader( object_of_class_Dataset, batch_size=n_data )))  # return the actual tuple (X,y)

#
# ~~~ Convert Tensors into a pytorch Dataset; from https://fmorenovr.medium.com/how-to-load-a-custom-dataset-in-pytorch-create-a-customdataloader-in-pytorch-8d3d63510c21
class convert_Tensors_to_Dataset(torch.utils.data.Dataset):
    #
    # ~~~ Define attributes
    def __init__( self, X_tensor, y_tensor, transform_list=None, **kwargs ):
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

#
# ~~~ Routine that defines the transform that sends an integer n to the (n+1)-th standard basis vector of R^{n_class} (for n in the range 0 <= n < n_class)
def hot_1_encode_an_integer( n_class, dtype=None ):
    def transform(y,dtype=dtype):
        y = y if isinstance(y,torch.Tensor) else torch.tensor(y)
        y = y.view(-1,1) if y.ndim==1 else y
        dtype = y.dtype if (dtype is None) else dtype
        return torch.zeros( size=(y.numel(),n_class), dtype=dtype ).scatter_( dim=1, index=y.view(-1,1), value=1 ).squeeze()       
    return transform


# todo for DataLoading onto GPU: https://stackoverflow.com/questions/65932328/pytorch-while-loading-batched-data-using-dataloader-how-to-transfer-the-data-t

#
# ~~~ A customizable sub-routine to be passed to the high level function `fit` defined below
def standard_train_step( model, data, loss_fn, optimizer, device, history, training_metrics, just, sig ):
    #
    # ~~~ Unpack data and move it to the desired device
    X = data[0].to(device)
    y = data[1].to(device)
    #
    # ~~~ Enter training mode and compute prediction error
    model.train()           # ~~~ set the model to training mode (e.g., activate dropout)
    loss = loss_fn(model(X),y)
    #
    # ~~~ Backpropagation
    loss.backward()         # ~~~ compute the gradient of loss
    optimizer.step()        # ~~~ apply the gradient to the model parameters
    #
    # ~~~ Record any user-specified metrics
    vals_to_print = { "loss": f"{loss.item():<{just}.{sig}f}" }  # ~~~ Here we'll store the value of any user-specified metrices, as well as adding them to history
    if training_metrics is not None:
        for key in training_metrics:
            value = training_metrics[key]( model=model, data=data, loss_fn=loss_fn, optimizer=optimizer )   # ~~~ pass the `required_kwargs` defined above
            history[key].append(value)
            vals_to_print[key] = f"{value:<{just}.{sig}f}"
    #
    # ~~~ Zero out the gradients, exit training mode, and return all metrics
    optimizer.zero_grad()   # ~~~ reset the gradient to zero so that we "start fresh" next time standard_train_step is called
    model.eval()            # ~~~ set the model to evaluation mode (e.g., deactivate dropout)
    return loss, history, vals_to_print

#
# ~~~ Standard training
def fit( model, training_batches, loss_fn, optimizer, train_step=standard_train_step, test_data=None, epochs=20, verbose=2, epochal_metrics=None, training_metrics=None, test_metrics=None, device=("cuda" if torch.cuda.is_available() else "cpu"), pbar_desc=None, just=3, sig=2, history=None ):
    #
    # ~~~ First, assert that training_metrics, test_metrics, and epochal_metrics are each a dictionary (if not None)
    assert isinstance(training_metrics,dict) or training_metrics is None
    assert isinstance(test_metrics,dict) or test_metrics is None
    assert isinstance(epochal_metrics,dict) or epochal_metrics is None
    #
    # ~~~ Require test data in order to use testing metrics (and prefer it for using epochal metrics)
    if test_data is None:
        if (test_metrics is not None) or (test_metrics is not None):
            raise ValueError("test_metrics requires test_data")
        if epochal_metrics is not None:
            my_warn("epocal_metrics will be applied on the most recent batch of training data, since test_data was not supplied")
    #
    # ~~~ Create a dictionary called history to store the value of the loss as we go, along with any user-supplied metrics
    history = { "loss":[] , "epoch":[] } if history is None else history
    required_kwargs = { "model", "data", "loss_fn", "optimizer" }
    names_already_used = set(history.keys())
    for metrics in ( training_metrics, test_metrics, epochal_metrics ):
        if metrics is not None:
            for key in metrics:
                assert key not in names_already_used
                history[key] = []
                names_already_used.add(key)
                vars = metrics[key].__code__.co_varnames[:metrics[key].__code__.co_argcount]    # ~~~ a list of text strings: the argumen
                accepts_arbitrary_kwargs = bool(metrics[key].__code__.co_flags & 0x08)          # ~~~ idk fam chat gpt came up with this one
                for this_required_kwarg in required_kwargs:
                    if not (accepts_arbitrary_kwargs or this_required_kwarg in vars):
                        name = metrics[key].__code__.co_name    # ~~~ a text string
                        vars = ','.join(vars)  
                        requirement = f"All metrics must support the keyword arguments {required_kwargs} (though they needn't be used in the body of the function)."
                        issue = f"Please modify the definition of {metrics[key].__code__.co_name} accordingly,"
                        suggestion = f"e.g., rewrite `def {name}({vars})` as `def {name}({vars},**kwargs)`" 
                        raise ValueError( requirement+" "+issue+" "+suggestion )
    #
    # ~~~ Train with a progress bar
    with support_for_progress_bars():
        #
        # ~~~ If verbose==1, then just make one long-ass progress bar for all epochs combined
        if verbose>0 and verbose<2:
            pbar = tqdm( desc=pbar_desc, total=epochs*len(training_batches), ascii=' >=' )
        #
        # ~~~ Cycle through the epochs
        for n in range(epochs):
            #
            # ~~~ If verbose==2, then create a brand new keras-style progress bar for each epoch
            if verbose>=2:
                title = f"Epoch {n+1}/{epochs}"
                if pbar_desc is not None:
                    title += f" ({pbar_desc})"
                pbar = tqdm( desc=title, total=len(training_batches), ascii=' >=' )
            #
            # ~~~ Cycle through all the batches into which the data is split up
            for data in training_batches:
                #
                # ~~~ Do the actual training (FYI: recording of any user-specified training metrics is intended to occor within the body of train_step; their values will be added to history *and* stored by themselves in vals_to_print which has a number of keys equal 1 greater than the number of user-specified training metrics (if None, then this dictionary's only key will be loss), and each key has a scalar value; the intend is to pass pbar.set_postfix(vals_to_print); in other words, vals_to_print already includes any training_metrics, but the loss and any test_metrics still need to be added to it)
                loss, history, vals_to_print = train_step(
                        model,
                        data,
                        loss_fn,
                        optimizer,
                        device,
                        history,
                        training_metrics,
                        just, sig
                    )
                #
                # ~~~ No matter what, always record this information (assume loss was already added to vals_to_print during train_step)
                history["loss"].append(loss.item())
                history["epoch"].append(n+1)
                #
                # ~~~ Regardless of verbosity level, still compute and record any test metrics (we may or may not later print them, depending on whether verbose>=2)
                if test_metrics is not None:
                    for key in test_metrics:
                        value = test_metrics[key]( model=model, data=test_data, loss_fn=loss_fn, optimizer=optimizer )  # ~~~ pass the `required_kwargs` defined above
                        history[key].append(value)
                        vals_to_print[key] = f"{value:<{just}.{sig}f}"
                #
                # ~~~ If 0<verbose<2, then print loss and nothing else
                if verbose>0 and verbose<2:
                    pbar.set_postfix( {"loss":f"{loss:<{just}.{sig}f}"} )
                #
                # ~~~ If instead verbose>=2, then print all training and test metrics in addition to the loss
                if verbose>=2:
                    pbar.set_postfix( vals_to_print, refresh=False )
                #
                # ~~~ Whether verbose==1 or verbose==2, update the progress bar each iteration
                if verbose>0:
                    pbar.update()
            #
            # ~~~ At the end of the epoch, if verbose==2, then finalize the message before closing the progress bar
            if verbose>=2:
                vals_to_print = { "loss":f"{loss:<{just}.{sig}f}" }  # ~~~ loss on the final iteration of this epoch
                if epochal_metrics is not None:
                    for key in epochal_metrics:
                        value = epochal_metrics[key]( model=model, data=(data if test_data is None else test_data), loss_fn=loss_fn, optimizer=optimizer )  # ~~~ pass the `required_kwargs` defined above
                        history[key].append(value)
                        vals_to_print[key] = f"{value:<{just}.{sig}f}"
                pbar.set_postfix( vals_to_print, refresh=False )
                pbar.close()
    #
    # ~~~ End
    if verbose>0 and verbose<2:
        pbar.close()
    return history


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