
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/quality_of_life

import sys
import numpy as np

from matplotlib import pyplot as plt
from plotly import graph_objects as go
from PIL import Image
from io import BytesIO
import os
from contextlib import contextmanager

from quality_of_life.my_base_utils import my_warn, process_for_saving, get_file_extension

this_is_running_in_colab = ('google.colab' in sys.modules)

class GifMaker:
    def __init__( self, path_or_name=os.path.join(os.getcwd(),"my_gif"), fps=30, ram_only=True ):
        self.frames = []
        self.ram_only = ram_only
        extension = get_file_extension(path_or_name)
        self.path_or_name = path_or_name.strip(extension)
        self.fps=fps
        if not ram_only:
            raise NotImplementedError("Automatic management of file names still needs to be debugged")
    def capture(self,multiple_exposure=False):
        buffer = BytesIO() if self.ram_only else None
        filename = process_for_saving(self.path_or_name)+".png" if not self.ram_only else None
        plt.savefig(buffer if self.ram_only else filename)
        self.frames.append(buffer.getvalue() if self.ram_only else filename)
        if not multiple_exposure:
            plt.close()
    def stable_capture(self):
        buffer = BytesIO() if self.ram_only else None
        filename = process_for_saving(self.path_or_name)+".png" if not self.ram_only else None
        plt.savefig(buffer if self.ram_only else filename)
        plt.clf()
        self.frames.append(buffer.getvalue() if self.ram_only else filename)
    def develop( self, destination, total_duration=None, fps=None, verbose=True ):
        #
        # ~~~ Process individual frames
        if self.ram_only:
            images = [ Image.open(BytesIO(buffer)) for buffer in self.frames ]
        else:
            images = [ Image.open(file) for file in self.frames ]
            # Clean up: Delete the individual PNG files
            for file in self.frames:
                os.remove(file)
        #
        # ~~~ Process destination path
        destination = self.path_or_name if destination is None else destination
        destination = os.path.join( os.getcwd(), destination ) if os.path.dirname(destination)=="" else destination
        destination = process_for_saving(destination.strip(".gif")+".gif")
        #
        # ~~~ Process frame rate
        fps = self.fps if fps is None else fps
        if total_duration is None:
            total_duration = len(self.frames)/fps
        else:
            fps = len(self.frames)/total_duration
        #
        # ~~~ Save the thing
        if verbose:
            print(f"Saving gif of length {total_duration:.3} sec. at {destination}")
        images[0].save( destination, save_all=True, append_images=images[1:], duration=int(1000/fps), loop=0 )


# @contextmanager
# def GifContext(path_or_name,ram_only):
#     gif = GifMaker(path_or_name,ram_only)
#     yield gif
#     gif.develop()

"""
# Example usage
from matplotlib import pyplot as plt
from quality_of_life.my_visualization_utils import GifMaker

gif = GifMaker()
for iteration in range(1, 101):
    # Your plotting logic
    _ = plt.plot(range(iteration), [i**2 for i in range(iteration)])
    _ = plt.xlabel('X-axis')
    _ = plt.ylabel('Y-axis')
    _ = plt.title(f'Iteration {iteration}')
    # Save the Matplotlib plot to an in-memory buffer or a file
    gif.capture()
    gif.clf()
    
gif.develop("test")
"""

#
# ~~~ Plot a line from slope and intercept
def abline( slope, intercept, *args, **kwargs ):   # ~~~ based off https://stackoverflow.com/a/43811762/11595884
    axes = plt.gca()
    original_xlim = axes.get_xlim()
    original_ylim = axes.get_ylim()
    x_vals = np.array(buffer(original_xlim,multiplier=1))
    y_vals = intercept + slope*x_vals
    plt.plot( x_vals, y_vals, *args, **kwargs )
    axes.set_xlim(original_xlim)
    axes.set_ylim(original_ylim)

#
#~~~ Compute [min-c,max+c] where c>0 is a buffer
def buffer(vector,multiplier=0.05):
    a = min(vector)
    b = max(vector)
    extra = (b-a)*multiplier
    return [a-extra, b+extra]

#
#~~~ Renders the image of a scatter plot overlaid with the graphs of one of more functions (I say "curve" because I'm remembering the R function `curve`)
def points_with_curves(
        x,
        y,
        curves,
        points_label = None,
        curve_colors = None,
        marker_size = None,
        marker_color = None,
        point_mark = None,
        curve_thicknesses = None,
        curve_labels = None,
        curve_marks = None,
        grid = None,
        title = None,
        xlabel = None,
        ylabel = None,
        xlim = None,
        ylim = None,
        crop_ylim = True,
        show = True,
        fig = "new",
        ax = "new",
        figsize = (6,6) if this_is_running_in_colab else None,
        model_fit = True    # default usage: the plot to be rendered is meant to visualize a model's fit
    ):
    #
    #~~~ Automatically set default values for several arguments
    n_curves = len(curves)
    if points_label is None:
        points_label = "Training Data"
    if marker_size is None:
        marker_size = 4 if len(x)<400 else max( 247/60-7/24000*len(x), 1.2)
    if marker_color is None:
        marker_color = "green"
    if point_mark is None:
        point_mark = 'o'
    if curve_colors is None:
        curve_colors = ( "green", "blue", "orange", "midnightblue", "red", "hotpink" )
    if xlim is None:
        xlim = buffer(x)
    if grid is None:
        grid = np.linspace( min(xlim), max(xlim), 1001 )
    if ylim is None and crop_ylim:
        lo,hi = 0,0
        extra = (hi-lo)*0.2
        ylim = buffer(y,multiplier=0.2)
    curve_thicknesses = [min(0.5,3/n_curves)]*n_curves if curve_thicknesses is None else curve_thicknesses[:n_curves]
    #
    #~~~ Facilitate my most common use case: the plot to be rendered is meant to visualize a model's fit
    if (curve_labels is None) and (curve_marks is None) and model_fit:
        curve_labels = ["Fitted Model","Ground Truth"] if n_curves==2 else [f"Fitted Model {i+1}" for i in range(n_curves-1)] + ["Ground Truth"]
        curve_marks = ["-"]*(n_curves-1) + ["--"]       # ~~~ make the last curve dashed
        curve_colors = curve_colors[:n_curves][::-1]    # ~~~ make the last curve green, which matches the default color of the dots
    elif model_fit:
        arg_vals, arg_names = [], []
        if curve_labels is not None:
            arg_names.append("curve_labels")
            arg_vals.append(str(curve_labels))
        if curve_marks is not None:
            arg_names.append("curve_marks")
            arg_vals.append(str(curve_marks))
        if not n_curves==2:
            arg_names.append("n_curves")
            arg_vals.append(str(n_curves))
        warning_msg = "the deafult `model_fit=True` is overriden by user_specified input(s):"
        for i,(name,val) in enumerate(zip(arg_names,arg_vals)):
            warning_msg += " " + name + "=" + val
            if i< len(arg_vals)-1:
                warning_msg += " |"
        # print(bcolors.HEADER)
        # warnings.warn(bcolors.WARNING+warning_msg+"\n"+bcolors.ENDC, stacklevel=2)
        my_warn(warning_msg)
    #
    #~~~ Other defaults and assertions
    curve_labels = [ f"Curve {i}" for i in range(n_curves) ] if curve_labels is None else curve_labels[:n_curves]
    curve_marks = [ '-' ]*n_curves if curve_marks is None else curve_marks[:n_curves]
    assert curve_thicknesses is not None
    assert curve_labels is not None
    assert curve_marks is not None
    assert grid is not None
    assert len(x) == len(y)
    assert len(curves) <= len(curve_labels)
    assert len(curves) <= len(curve_colors)
    #        
    #~~~ Do the thing
    fig,ax = plt.subplots(figsize=figsize) if (fig=="new" and ax=="new") else (fig,ax)   # supplied by user in the latter case
    ax.plot( x, y, point_mark, markersize=marker_size, color=marker_color, label=points_label )
    for i in range(n_curves):
        ax.plot( grid, curves[i](grid), curve_marks[i], curve_thicknesses[i], color=curve_colors[i], label=curve_labels[i] )
    #
    #~~~ Further aesthetic configurations
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid()
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    #
    #~~~ The following lines replace `plt.legend()` to avoid duplicate labels; source https://stackoverflow.com/a/13589144
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = list(set(labels))  # Get unique labels
    by_label = {}   # Create a dictionary to store handles and line styles for each unique label
    for label in unique_labels:
        indices = [i for i, x in enumerate(labels) if x == label]  # Find indices for each label
        handle = handles[indices[0]]  # Get the handle for the first occurrence of the label
        line_style = handle.get_linestyle()  # Get the line style
        by_label[label] = (handle, line_style)  # Store handle and line style
    legend_handles = [by_label[label][0] for label in by_label]
    legend_labels = [f"{label}" for label in by_label]  # Include line style in label
    plt.legend(legend_handles,legend_labels)
    if show:
        fig.tight_layout()
        plt.show()
    else:
        return fig, ax

#
# ~~~ A helper routine for plotting (and thus comparing) results
def side_by_side_prediction_plots(
            x,
            y,
            true_fun,
            pred_a,
            pred_b,
            axatitle = "One Model",
            axbtitle = "Another Model",
            other_x = None,
            other_y = None,
            figsize = (12,6) if this_is_running_in_colab else None,
            figtitle = None,
            **kwargs
       ):
    #
    # ~~~ Use the same points for both images unless specified otherwise
    other_x = x if other_x is None else other_x
    other_y = y if other_y is None else other_y
    #
    # ~~~ Create a window with space for two images
    fig,(ax_a,ax_b) = plt.subplots( 1, 2, figsize=figsize )
    #
    # ~~~ Create the left image
    fig,ax_a = points_with_curves(
            x = x, 
            y = y, 
            curves = (pred_a,true_fun), 
            title = axatitle, 
            show = False, 
            fig = fig, 
            ax = ax_a,
            **kwargs
        )
    #
    # ~~~ Create the left image
    fig,ax_b = points_with_curves(
            x = other_x,
            y = other_y,
            curves = (pred_b,true_fun),
            title = axbtitle,
            show = False,
            fig = fig,
            ax = ax_b,
            **kwargs
        )
    #
    # ~~~ Print the combined window containing both images
    if figtitle is not None:
        plt.suptitle(figtitle)
    fig.tight_layout()
    plt.show()

#
# ~~~ Create a surface plot of f, assuming Z is the len(x)-by-len(y) matrix with Z[i,j]=f(x[i],y[j])
def matrix_surf(x,y,Z):
    go.Figure(go.Surface( x=x, y=y, z=Z )).show()

# #
# # ~~~ Return the len(x)-by-len(y) matrix Z matrix defined by Z[i,j] = f(x[i],y[j])
# def apply_on_cartesian_product(f,x,y):
#     X, Y = np.meshgrid(x,y)
#     cartesian_product = np.column_stack((X.flatten(), Y.flatten())) # ~~~ the result is basically just a rearranged version of list(itertools.product(x,y))
#     return f(cartesian_product).reshape(X.shape).T

# #
# # ~~~ Wrap it in a neat package
# def func_surf(x,y,f):
#     matrix_surf( x, y, apply_on_cartesian_product(f,x,y) )

# #
# # ~~~ Wrap it in a neat package
# def basic_surf( f, xlim, ylim, res=501 ):
#     x = np.linspace( xlim[0], xlim[-1], res )
#     y = np.linspace( ylim[0], ylim[-1], res )
#     func_surf(x,y,f)



# x = np.linspace(0,5,1001)
# y = np.linspace(1,2,301)
# f = lambda matrix: np.sin(np.sum(matrix**2,axis=1))
# func_surf(x,y,f)