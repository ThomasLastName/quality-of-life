
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/quality_of_life

import warnings
from quality_of_life.my_plt_utils import *
from quality_of_life.my_plotly_utils import *

def complain():
    warnings.warn( "The sub-module `my_visualization_utils` is deprecated. To reduce dependencies, it has been divided into `my_plt_utils`, `my_plotly_utils`, and `my_scipy_utils`; see also https://github.com/ThomasLastName/quality-of-life/issues/3", DeprecationWarning )
    return

complain()
#