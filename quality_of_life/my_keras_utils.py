
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/quality_of_life

import sys
import inspect
import tensorflow as tf


def keras_seed(semilla):    
    tf.random.set_seed(semilla)
    tf.keras.utils.set_random_seed(semilla)
    if "numpy" in sys.modules.keys():
        sys.modules["numpy"].random.seed(semilla)
    if "random" in sys.modules.keys():
        sys.modules["random"].seed(semilla)


### ~~~
## ~~~ Builds a Keras dense sequential model with specified input and output dimensions, specified number of hidden layers
## ~~~ and widths of each hidden layer, and specified activation functions.
### ~~~

def make_keras_network( num_inputs, num_outputs, hidden_layers=[24,18,8,8,4], activations=None, kernel_initializer='he_normal') :
    ### ~~~
    ## ~~~  Convenience features
    ### ~~~
    try:
        hidden_layers = list(hidden_layers)
    except Exception as e:
        if type(e) is TypeError:
            hidden_layers = list([hidden_layers])
        else:
            raise
    if activations is None:
        activations = ['tanh' for layer in hidden_layers]
    if not isinstance(activations,list):
        activations = [activations for layer in hidden_layers]
    ### ~~~
    ## ~~~  Safety features
    ### ~~~
    #
    #   todo
    #
    ### ~~~
    ## ~~~  Do the thing
    ### ~~~
    model = tf.keras.Sequential()
    #
    #~~~ add input layer
    model.add(tf.keras.layers.Input(shape=(num_inputs,)))
    #
    #~~~ add hidden layers with specified activations
    for (layer_size, activation) in zip(hidden_layers,activations):
        model.add(tf.keras.layers.Dense(layer_size, activation=activation, kernel_initializer=kernel_initializer))
    #
    #~~~ add output layer
    model.add(tf.keras.layers.Dense(num_outputs, kernel_initializer=kernel_initializer))
    return model


def lazy_keras_training(
        model,
        x_train = None,
        y_train = None,
        x_val = None,
        y_val = None,
        loss = "mse", #tf.keras.losses.MeanSquaredError()
        optimizer="adam",
        verbose=2,
        batch_size = 64,
        epochs = 20,
        *args,
        **kwargs
    ):
    ### ~~~
    ## ~~~ Call the compile method
    ### ~~~
    model.compile(
            optimizer = optimizer,
            loss=loss
        )
    ### ~~~
    ## ~~~ Window dressing
    ### ~~~
    if verbose==2:
        model.summary()
    if verbose>0:
        print("")
        print("    Now training the model.")
        print("")
    ### ~~~
    ## ~~~ Access global variables out of laziness
    ### ~~~
    #
    # ~~~ Retrive the gloabl variables of the setting in which lazy_keras_training is called, as they told me *not* to do in the comments of https://stackoverflow.com/q/77761711/11595884
    a,b,c,d = False, False, False, False    # flags
    gloabl_variables = inspect.currentframe().f_back.f_globals
    if "x_train" in gloabl_variables.keys() and x_train is None:
        x_train = gloabl_variables["x_train"]
        a = True
    if "y_train" in gloabl_variables.keys() and y_train is None:
        y_train = gloabl_variables["y_train"]
        b = True
    if "x_val" in gloabl_variables.keys() and x_val is None:
        x_val = gloabl_variables["x_val"]
        c = True
    if "y_val" in gloabl_variables.keys() and y_val is None:
        y_val = gloabl_variables["y_val"]
        d = True
    ### ~~~
    ## ~~~ Coercion of variable types
    ### ~~~
    x_train = tf.convert_to_tensor(x_train) if type(x_train).__module__=="numpy"    else x_train
    y_train = tf.convert_to_tensor(y_train) if type(y_train).__module__=="numpy"    else y_train
    x_val = tf.convert_to_tensor(x_val)     if type(x_val).__module__=="numpy"      else x_val
    y_val = tf.convert_to_tensor(y_val)     if type(y_val).__module__=="numpy"      else y_val
    ### ~~~
    ## ~~~ Call (and return the result of calling) the fit method
    ### ~~~
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_data=(x_val,y_val),
        *args,
        **kwargs
        )
    ### ~~~
    ## ~~~  Clear memory, I guess?
    ### ~~~
    if a:
        del x_train
    if b:
        del y_train
    if c:
        del x_val
    if d:
        del y_val
    ### ~~~
    ## ~~~  Done
    ### ~~~
    return history.history


