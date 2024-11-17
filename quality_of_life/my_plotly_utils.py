
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/quality_of_life

import warnings
import numpy as np
from plotly import graph_objects as go

from quality_of_life.my_base_utils import my_warn, buffer
from quality_of_life.my_numpy_utils import apply_on_cartesian_product
from quality_of_life.my_scipy_utils import extend_to_grid

#
# ~~~ Create a surface plot of f, assuming Z is the len(x)-by-len(y) matrix with Z[i,j]=f(x[i],y[j])
def matrix_viz( x, y, Z, graph_object, verbose=True, **kwargs,  ):
    #
    # ~~~ Do the plotly equivalent of plt.plot
    fig = go.Figure(graph_object( x=x, y=y, z=Z ))
    #
    # ~~~ Further figure settings
    try:
        if len(kwargs)>0:
            fig.update_layout(**kwargs) # ~~~ acceptable kwargs can be found at https://plotly.com/python/reference/layout/
        fig.show()
        if verbose:
            print("Image opened in browser.")
    except:
        if verbose:
            my_warn("Certain kwargs are not accepted by `plotly.graph_objects.Figure`. Instead the figure will be returned for manual setting.")
        return fig

#
# ~~~ Create a surface plot of f, assuming the vector z has len(z)==len(x)==len(y), z[k]=f(x[k],y[k])
def vector_viz( x, y, z, graph_object=go.Surface, verbose=True, extrapolation_percent=0.05, res=301, **kwargs ):
    #
    # ~~~ Interpolate/extrapolate onto the mesh
    X, Y, Z = extend_to_grid( x, y, z, extrapolation_percent=extrapolation_percent, res=res )
    #
    # ~~~ Render the interpolated surface
    fig = go.Figure( graph_object(x=X,y=Y,z=Z) )
    #
    # ~~~ Further figure settings
    try:
        fig.update_layout(**kwargs) # ~~~ acceptable kwargs can be found at https://plotly.com/python/reference/layout/
        fig.show()
        if verbose:
            print("Image opened in browser.")
    except:
        if verbose:
            my_warn("Certain kwargs are not accepted by `plotly.graph_objects.Figure`. Instead the figure will be returned for manual setting.")
        return fig

#
# ~~~ Return the surface plot of a function f = lambda xy_pairs: f(xy_pairs) on the Cartesian product grid x \times y (calls `matrix_surf`)
def cp_surf( x, y, f, graph_object=go.Surface, **kwargs ):
    #
    # ~~~ Form the matrix Z[i,j] = f([x[i],y[j]])
    try:
        from quality_of_life.my_torch_utils import apply_on_cartesian_product as torch_apply_on_cartesian_product
        Z = torch_apply_on_cartesian_product( f, x, y )
        Z = Z.cpu().detach().numpy()
    except:
        Z = apply_on_cartesian_product( f, x, y )
    #
    # ~~~ Call the originally defined routine
    return matrix_viz( x, y, Z, graph_object=graph_object, **kwargs )

#
# ~~~ Return the surface plot of a function f = lambda xy_pairs: f(xy_pairs) on the cell xlim \times ylim (calls `func_surf` which calls `matrix_surf`)
def cell_surf( f, xlim, ylim, graph_object=go.Surface, res=301, **kwargs ) :
    try:
        import torch
        _ = f(torch.randn(20,2))
        x = torch.linspace( xlim[0], xlim[-1], res )
        y = torch.linspace( ylim[0], ylim[-1], res )
    except:
        x = np.linspace( xlim[0], xlim[-1], res )
        y = np.linspace( ylim[0], ylim[-1], res )
    return cp_surf( x, y, f, graph_object=graph_object, **kwargs )

#
# ~~~ Temporary alias: `func_surf` is being renamed to `cp_surf`
def func_surf(*args,**kwargs):
    warnings.warn( "`func_surf` is being renamed to `cp_surf`. Please use the new name instead of the old one.", DeprecationWarning )
    return cp_surf(*args,**kwargs)

#
# ~~~ Temporary alias: `basic_surf` is being renamed to `cell_surf`
def basic_surf(*args,**kwargs):
    warnings.warn( "`basic_surf` is being renamed to `cell_surf`. Please use the new name instead of the old one.", DeprecationWarning )
    return cp_surf(*args,**kwargs)


# x = np.linspace(0,5,1001)
# y = np.linspace(1,2,301)
# f = lambda matrix: np.sin(np.sum(matrix**2,axis=1))
# cp_surf(x,y,f)
# cp_surf(x,y,f,graph_object=go.Heatmap)

