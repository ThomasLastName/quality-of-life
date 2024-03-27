
import sys
import math
import torch
from tqdm import tqdm
from time import time as now
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
    vals_to_print = { "loss": f"{loss.item():<{just}.{sig}f}" }  # ~~~ here we'll store the value of any user-specified metrices, as well as adding them to history
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
def fit( model, training_batches, loss_fn, optimizer, train_step=standard_train_step, test_data=None, epochs=20, verbose=2, training_metrics=None, test_metrics=None, epochal_training_metrics=None, epochal_test_metrics=None, device=("cuda" if torch.cuda.is_available() else "cpu"), pbar_desc=None, just=3, sig=2, history=None ):
    #
    # ~~~ First safety feature: assert that training_metrics, test_metrics, and epochal_metrics are each a dictionary (if not None)
    for metrics in ( training_metrics, test_metrics, epochal_training_metrics, epochal_test_metrics ):
        assert isinstance(metrics,dict) or (metrics is None)
    #
    # ~~~ Second safety feature: assert that no two dictionaries use the same key name
    unique_metric_key_names = set()
    repeat_metric_key_names = list()
    for metrics in ( training_metrics, test_metrics, epochal_training_metrics, epochal_test_metrics ):
        if metrics is not None:
            unique_metric_key_names = unique_metric_key_names | set(metrics.keys())
            repeat_metric_key_names = repeat_metric_key_names + list(metrics.keys())
    if len(repeat_metric_key_names)==len(unique_metric_key_names):
        metric_key_names = repeat_metric_key_names
    else:
        raise ValueError("key conflict: metrics in different dictionaries are referred to by the same key")
    #
    # ~~~ Third safety feature: assert that "loss", "epoch", and "time" are not the keys for any user-specified metric
    if "loss" in metric_key_names or "epoch" in metric_key_names or "time" in metric_key_names:
        raise ValueError("key conflict: the keys 'loss' and 'epoch' and 'time' are reserved and may not be used as the key names for a user-specified metric")
    #
    #
    # ~~~ Fourth safety feature: require test data in order to use testing metrics
    if test_data is None and ( (test_metrics is not None) or (epochal_test_metrics is not None) ):
        raise ValueError("test_metrics requires test_data")
    #
    # ~~~ Create (or contine) the dictionary called `history` to store the value of the loss, along with any user-supplied metrics
    no_history = (history is None)  # ~~~ whether or not we were given some priori history
    if no_history:
        #
        # ~~~ If we were *not* given any prior history, then create a blank dictionary called `hisory`
        history = { "loss":[] , "epoch":[], "time":[] }
        for key in metric_key_names:
            history[key] = []
    else:
        #
        # ~~~ If we *were* given a prior history, check that it is a dictionary
        if not isinstance(history,dict):
            raise ValueError("A supplied `history` argument must be a dictionary")
        #
        # ~~~ Check that the given history has keys called "loss" and "epoch" and "time"
        if not ("loss" in history.keys()) and ("epoch" in history.keys()) and ("time" in history.keys()):
            raise ValueError("keyerror: a supplied `history` dictionary must include keys for 'loss' and 'epoch'")
        #
        # ~~~ Check that the fileds of the given history are lists of compatible lengthts
        lengths = []
        for key in history:
            assert isinstance( history[key], list ), "Each field in `history` should be a list"
            lengths.append( len(history[key]) )
        lenghts_are_good = len(set(lengths))<=2
        lenghts_are_good = lenghts_are_good and max(lengths)==len(history["loss"])      # ~~~ one acceptable length is the number of iterations (for non-epochal metrics)
        if len(set(lengths))==2:
            lenghts_are_good = lenghts_are_good and min(lengths)==max(history["epoch"]) # ~~~ another acceptable length is the number of epochs (for epochal metrics)
        if not lenghts_are_good:
            for key in history:
                print(f"key '{key}' has length {len(history[key])}")
            raise ValueError("There is an inconsistency in the lenghts of the lists in the dictionary `history`. Each should either be the number of iterations (the length of history['loss']) or the number of epochs (the maximum of history['epoch'])")
        #
        # ~~~ Attempt to catch/correct the pratcice of changing metrics between calls to `fit` which is prone to procduce data incosistencies
        for j,metrics in enumerate(( training_metrics, test_metrics, epochal_training_metrics, epochal_test_metrics )):
            if metrics is not None:
                for key in metrics:
                    metric_name = f"`{metrics[key].__code__.co_name}`" if hasattr( metrics[key], "__code__" ) else f"'{key}'"
                    this_metric_is_only_every_epoch = (j>1)
                    #
                    # ~~~ If there are any metrics for which we have no history, retroactively populate the historical data on those metrics (populate with zeros)
                    if key not in history.keys():
                        my_warn(f"User-supplied history does not contain a key matching the key '{key}' of user-supplied metric {metric_name}. Historical data for this key value will be populated with zeros.")
                        appropriate_length = max(history["epoch"]) if this_metric_is_only_every_epoch else len(history["loss"])
                        history[key] = appropriate_length*[0]
                    #
                    # ~~~ If we have historical data on one of the metrics we aren't going to record during the upcoming round of training, compare the historical and upcoming frequency of records
                    else:
                        future_data_will_be_only_every_epoch = (j>1)
                        historical_data_was_only_every_epoch = ( len(history[key])==max(history["epoch"]) and not len(history[key])==len(history["loss"]) )
                        if not historical_data_was_only_every_epoch==future_data_will_be_only_every_epoch:  # ~~~ if not either both true or both false, i.e., if the frequencies don't match
                            freq_of_historical_data = "epoch" if historical_data_was_only_every_epoch else "iteration"
                            freq_of_future_data     = "epoch" if future_data_will_be_only_every_epoch else "iteration"
                            my_warn(f"IMPORTANT! User-supplied historical data on key '{key}' was avaliable for every {freq_of_historical_data}. However, it is to be collected every {freq_of_future_data} going forward. The change in the frequency at which this metric is measured will result in a data anomaly in the history dictionary. Please handle this after training is complete (e.g., by constant interpolation).")
        #
        # ~~~ If there is historical data on any metric that we will *not* track in the upcoming round of training, then extend the historical data by populating with zeros
        for key in history:
            if (key not in metric_key_names) and (key not in ("loss","epoch","time")):
                n_data_per_epoch_on_this_key = len(history[key])/max(history['epoch'])
                assert n_data_per_epoch_on_this_key==int(n_data_per_epoch_on_this_key), f"Why isn't the length of data for {key} divisible by the number of epochs!?"
                history[key] += int(n_data_per_epoch_on_this_key)*[0]*epochs
                my_warn(f"User-supplied history contains a key '{key}' which does not match the key of any user-supplied metric. The historical data has been extended by populating it with zeros.")
    #
    # ~~~ Validate that all metrics meet some assumptions upon which the present code is positted: namely, that metric keys are unique, and all keys support the `required_kwargs`
    required_kwargs = { "model", "data", "loss_fn", "optimizer" }   # ~~~ below, we'll call `metric( model=model, data=data, loss_fn=loss_fn, optimizer=optimizer )`
    #
    # ~~~ For all user-supplied metrics...
    for j,metrics in enumerate(( training_metrics, test_metrics, epochal_training_metrics, epochal_test_metrics )):
        #
        # ~~~ ... (if any) ...
        if metrics is not None:
            for key in metrics:
                #
                # ~~~ ... check that the function metrics[key] accepts as a keyword argument each of the `required_kwargs` (idk fam chat gpt came up with the next two lines)
                metric_name = f"`{metrics[key].__code__.co_name}`" if hasattr( metrics[key], "__code__" ) else "metric_function"
                try:
                    #
                    # ~~~ Attempt a generic workaround in order to allow metrics not only to be functions but, also, potentially to be any object with a __call__ method
                    metric_name = f"`{metrics[key].__code__.co_name}`" if hasattr( metrics[key], "__code__" ) else "metric_function"
                    if not hasattr( metrics[key], "__code__" ):
                        metrics[key].__code__ = metrics[key].__call__.__code__
                    vars = metrics[key].__code__.co_varnames[:metrics[key].__code__.co_argcount]    # ~~~ a list of text strings: the names of the arguments that metric_name==metrics[key] accepts
                    accepts_arbitrary_kwargs = bool(metrics[key].__code__.co_flags & 0x08)          # ~~~ whether or not metrics[key] was defined with a `**kwargs` catchall
                    #
                    # ~~~ Specifically, for each of the keyword arguments `required_kwargs` that every metric is assumed to accept, ...
                    for this_required_kwarg in required_kwargs:
                        #
                        # ~~~ ... if this metric does not aaccept that keyword argument, ...
                        if not (accepts_arbitrary_kwargs or (this_required_kwarg in vars)):
                            #
                            # ~~~ ... then write and raise an error message complaining about it
                            vars = ','.join(vars)   # ~~~ a single text string: the names of the arguments that his metric accepts which we concatenate and separate by commas
                            requirement = f"All metrics must support the keyword arguments {required_kwargs} (though they needn't be used in the body of the metric)."
                            violation = f"Please modify the definition of {metric_name} to accept the keyword argument {this_required_kwarg}"
                            suggestion = f", e.g., replace `def {metric_name}({vars})` by `def {metric_name}({vars},**kwargs)`" 
                            raise ValueError( requirement+" "+violation+" "+suggestion )
                except:
                    my_warn(f"Unable to read the arguments of metrics. Please, be aware that an error will occur if a metric does not suppport the arguments {str(required_kwargs)}.")
    #
    # ~~~ Train with a progress bar
    with support_for_progress_bars():
        #
        # ~~~ If verbose==1, then just make one long-ass progress bar for all epochs combined
        if verbose>0 and verbose<2:
            pbar = tqdm( desc=pbar_desc, total=epochs*len(training_batches), ascii=' >=' )
        #
        # ~~~ Regardless of verbosity level, cycle through the epochs
        n_epochs_completed_berforehand = 0 if no_history else max(history["epoch"])
        for n in range(epochs):
            #
            # ~~~ If verbose==2, then create a brand new keras-style progress bar at the beginning of each epoch
            if verbose>=2:
                title = f"Epoch {n+1+n_epochs_completed_berforehand}/{epochs+n_epochs_completed_berforehand}"
                if pbar_desc is not None:
                    title += f" ({pbar_desc})"
                pbar = tqdm( desc=title, total=len(training_batches), ascii=' >=' )
            #
            # ~~~ Cycle through all the batches into which the data is split up
            for j,data in enumerate(training_batches):
                this_is_final_iteration_of_this_epoch = (j+1==len(training_batches))
                #
                # ~~~ Do the actual training (FYI: recording of any user-specified training metrics is intended to occor within the body of train_step; their values will be added to history *and* stored by themselves in vals_to_print which has a number of keys equal 1 greater than the number of user-specified training metrics (if None, then this dictionary's only key will be loss), and each key has a scalar value; the intent is to pass pbar.set_postfix(vals_to_print); in other words, vals_to_print already includes any training_metrics, but the loss; any test_metrics still need to be added to it)
                loss, history, vals_to_print = train_step(
                        model,
                        data,
                        loss_fn,
                        optimizer,
                        device,
                        history,
                        {**training_metrics,**epochal_training_metrics} if (this_is_final_iteration_of_this_epoch and (epochal_training_metrics is not None)) else training_metrics,
                        just, sig
                    )
                #
                # ~~~ No matter what, always record this information (FYI: it is assumed that loss.item() was already added to vals_to_print during train_step)
                history["loss"].append(loss.item())
                history["epoch"].append( n+1 + n_epochs_completed_berforehand )
                history["time"].append(now())
                #
                # ~~~ Regardless of verbosity level, still compute and record any test metrics (we may or may not print them, depending on whether or not verbose>=2)
                if test_metrics is not None:
                    for key in test_metrics:
                        value = test_metrics[key]( model=model, data=test_data, loss_fn=loss_fn, optimizer=optimizer )  # ~~~ pass the `required_kwargs` defined above
                        history[key].append(value)
                        vals_to_print[key] = f"{value:<{just}.{sig}f}"
                #
                # ~~~ If 0<verbose<2, then print loss and nothing else
                if verbose>0 and verbose<2:
                    pbar.set_postfix( {"loss":f"{loss.item():<{just}.{sig}f}"} )
                #
                # ~~~ If instead verbose>=2, then print all training and test metrics in addition to the loss
                if verbose>=2:
                    pbar.set_postfix( vals_to_print, refresh=False )
                #
                # ~~~ Whether verbose==1 or verbose==2, update the progress bar each iteration
                if verbose>0:
                    pbar.update()
            #
            # ~~~ At the end of the epoch, regardless of verbosity level, compute any user-specified epochal_test_metrics (the epochal_training_metrics were already computed, and added to both `history` and `vals_to_print` during train_step when `this_is_final_iteration_of_this_epoch`)
            if epochal_test_metrics is not None:
                for key in epochal_test_metrics:
                    value = epochal_test_metrics[key]( model=model, data=test_data, loss_fn=loss_fn, optimizer=optimizer )  # ~~~ pass the `required_kwargs` defined above
                    history[key].append(value)
                    vals_to_print[key] = f"{value:<{just}.{sig}f}"
            #
            # ~~~ At the end of the epoch, if verbose>=2, then finalize the message before closing the progress bar
            if verbose>=2:
                #
                # ~~~ Discard any user-specified training_metrics or training_metrics from vals_to_print (the "loss" will remain, as it is not user-specified) 
                for metrics in (training_metrics,test_metrics):
                    if metrics is not None:
                        for key in metrics:
                            _ = vals_to_print.pop(key)
                #
                # ~~~ Print the final loss of this iteration, along with any use-specified epochal train/test metrics
                pbar.set_postfix( vals_to_print, refresh=False )
                #
                # ~~~ Shut off the progress bar
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