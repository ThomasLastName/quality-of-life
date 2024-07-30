
import numpy as np
from quality_of_life.my_base_utils import buffer
from scipy.interpolate import griddata

#
# ~~~ Given any vectors x and y not on a grid, and a vector z[j]=f(x[j],y[j]), interploate/extrapolate to square mesh-grids X and Y
def extend_to_grid( x, y, z, extrapolation_percent=0.05, res=501, method="cubic" ):
    #
    # ~~~ Infer an xlim and ylim
    x_lo, x_hi = buffer( x, multiplier=extrapolation_percent )
    y_lo, y_hi = buffer( y, multiplier=extrapolation_percent )
    #
    # ~~~ Create a mesh for interpolation
    X, Y = np.meshgrid(
        np.linspace( x_lo, x_hi, res ),
        np.linspace( y_lo, y_hi, res )
    )
    #
    # ~~~ Interpolate onto the mesh and render the interpolated surface
    Z = griddata( np.column_stack([x,y]), z, (X,Y), method=method )
    #
    # ~~~ In its new processed form, `conventional_plotting_function(X,Y,Z)` should hopefully work
    return X,Y,Z
