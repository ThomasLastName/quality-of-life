# QAULITY OF LIFE HELPER ROUTINES
These are some helper routines that I want to be able to load without rewriting them every time. For instance, now that I've found a way in which I prefer to format my matplotlib figures, I don't want to have to copy and paste the corresponding 10 lines of code every time. Instead, I prefer to define a callable function that implements this routine. This repo is full of such callable functions for tasks which I want to do often in many different projects.


---

# Demo: Making .gif's

[https://colab.research.google.com/drive/10sOSQChyJrrajtvnBRkfhUHAI-tNO6zQ?usp=sharing](https://colab.research.google.com/drive/10sOSQChyJrrajtvnBRkfhUHAI-tNO6zQ?usp=sharing)

---

# Prerequisites for Using This Code

**Currently, installation requires that you have git installed on your machine!** (see also [#7](https://github.com/ThomasLastName/quality-of-life/issues/7))

This repo depends only on some standard libraries. The ones installed automatically are `numpy`, `scipy`, `matplotlib`, `plotly`, `tqdm`, and `requests`. However, the more advanced dependencies are left to the user to install manually. For instance, the sub-module `my_torch_utils` only works if you have pytorch installed, the sub-module `my_openai_utils` only works if you have openai installed, the sub-module `my_cvx_utils` only works if you have cvxpy installed, etc.

As a result, only the following work "out of the box" (see **Installation**) _without_ the need to manually install dependencies:
 - `ansi.py`
 - `my_base_utils.py`
 - `my_numpy_utils.py`
 - `my_plotly_utils.py`
 - `my_plt_utils.py`
 - `my_scipy_utils.py`
 - `my_visualization_utils.py` ([deprecated](https://github.com/ThomasLastName/quality-of-life/issues/3))

---

# Installation/Upgrading

**Currently, installation requires that you have git installed on your machine!** (see also [#7](https://github.com/ThomasLastName/quality-of-life/issues/7))

For general users, the same command `pip install --upgrade git+https://github.com/ThomasLastName/quality-of-life.git` can be used for both installing the code and updating the code. However, as noted above, only certain sub-modules will function "out of the box" (see **Prerequisites for Using This Code**). Failing that, try following the developer installation instructions (next paragraph)

For developers, I recommend instead cloning the repo as normal, navigating to the root directory of the repo, and then using the command `pip install -e .` (the `-e` flag stands for "editable", and the `.` indicates the current working directory). This way, "if you update the code from Github [or locally], your installation will automatically take those changes into account without requiring re-installation" (explanation borrowed from [these docs](https://sepia-lanl.readthedocs.io/en/latest/)).

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


Additionally, in the source code for `warnings.py` (in Windows, this is also in `Lib` folder), I recommend modifying the definition of `_showwarnmsg_impl` as follows so that warnings will print in yellow (if anyone knows of a way to do this without modifying the source code, LMK please).

```
#
# ~~~ Yellow warnings
def _showwarnmsg_impl(msg):
    # tom was here
    import os, platform
    if platform.system()=="Windows":    # platform.system() returns the OS python is running o0n | see https://stackoverflow.com/q/1854/11595884
        os.system("color")              # Makes ANSI codes work | see Szabolcs' comment on https://stackoverflow.com/a/15170325/11595884
    class bcolors:                      # https://stackoverflow.com/a/287944/11595884
        WARNING = '\033[93m'
        ENDC = '\033[0m'
    file = msg.file
    if file is None:
        file = sys.stderr
        if file is None:
            # sys.stderr is None when run with pythonw.exe:
            # warnings get lost
            return
    text = bcolors.WARNING+_formatwarnmsg(msg)+bcolors.ENDC     # Tom was here
    try:
        file.write(text)
    except OSError:
        # the file (probably stderr) is invalid - this warning gets lost.
        pass

```
