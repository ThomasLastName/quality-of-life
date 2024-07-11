# QAULITY OF LIFE HELPER ROUTINES
These are some helper routines that I want to be able to load without rewriting them every time. For instance, now that I've found a way in which I prefer to format my matplotlib figures, I don't want to have to copy and paste the corresponding 10 lines of code every time. Instead, I prefer to define a callable function that implements this routine. This repo is full of such callable functions for tasks which I want to do often in many different projects.


---

# Demo: Making .gif's

[https://colab.research.google.com/drive/10sOSQChyJrrajtvnBRkfhUHAI-tNO6zQ?usp=sharing](https://colab.research.google.com/drive/10sOSQChyJrrajtvnBRkfhUHAI-tNO6zQ?usp=sharing)

---

# Prerequisites for Using This Code

**Currently, installation requires that you have git installed on your machine!** (see also [#7](https://github.com/ThomasLastName/quality-of-life/issues/7))

This repo depends only on some standard libraries. The ones installed automatically are `numpy`, `scipy`, `matplotlib`, `plotly`, and `tqdm`. However, the more advanced dependencies are left to the user to install manually. For instance, the sub-module `my_torch_utils` only works if you have pytorch installed, the sub-module `my_openai_utils` only works if you have openai installed, the sub-module `my_cvx_utils` only works if you have cvxpy installed, etc.

As a result, only the following work "out of the box" (see **Installation**) _without_ the need to manually install dependencies:
 - `ansi.py`
 - `my_base_utils.py`
 - `my_numpy_utils.py`
 - `my_plotly_utils.py`
 - `my_plt_utils.py`
 - `my_scipy_utils.py`
 - `my_visualization_utils.py` ([deprecated](https://github.com/ThomasLastName/quality-of-life/issues/3))

---

# Installation

Have git installed on your machine. The same command `pip install --upgrade git+https://github.com/ThomasLastName/quality-of-life.git` can be used for both installing the code and updating the code. However, as noted above, only certain sub-modules will function "out of the box" (see **Prerequisites for Using This Code**).

Failing that, try cloning this repo to wherever you want, then adding the directory of the repo to the PATH for your environment (or system) and installing all dependencies manually. I think this can be accomplished in 4 steps? First `cd wherever/the/hell`, then `git clone https://github.com/ThomasLastName/quality-of-life.git`, next `cd quality-of-life`, and finally `pip install -e .`? If you install this way, you'll also need to manually install _all_ of the dependencies.

---

# Documentation

Boy, I sure should write some!

---

# Usage
That's up to you! However, please credit me with a comment along the lines of `# ~~~ Tom wrote these; maintained at https://github.com/ThomasLastName/quality-of-life` in your code, if you use these!

---

## Recommend Application: Green Ouput, Yellow Warnings, and Red Errors (Windows)

(For non-Windows, replace `Lib` with any directory on the PATH) After installing this code (see above), I recommend creating a `usercustomize.py` file in your `Lib` folder containing the following code
```
#
#~~~ Green outputs
from quality_of_life.my_base_utils import colored_console_output
colored_console_output(warn=False)

#
#~~~ Red errors
from quality_of_life.my_base_utils import red_errors
red_errors()
```

or, if you already have a `usercustomize.py` file, consider adding the above code to it.


~Additionally, in the source code for `warnings.py` (also in your `Lib` folder), I recommend modifying the definition of `_showwarnmsg_impl` as follows so that warnings will print in yellow~ (this is deprecated, because it depends on having the deprecated `quality_of_life` inside of a parent directory that is on the path; this is [issue #6](https://github.com/ThomasLastName/quality-of-life/issues/6))

```
#
# ~~~ Yellow warnings
def _showwarnmsg_impl(msg):
    from quality_of_life.ansi import bcolors
    file = msg.file
    if file is None:
        file = sys.stderr
        if file is None:
            # sys.stderr is None when run with pythonw.exe:
            # warnings get lost
            return
    text = bcolors.WARNING+_formatwarnmsg(msg)+bcolors.ENDC
    try:
        file.write(text)
    except OSError:
        # the file (probably stderr) is invalid - this warning gets lost.
        pass
```
