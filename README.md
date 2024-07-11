# QAULITY OF LIFE HELPER ROUTINES
These are some helper routines that I want to be able to load without rewriting them every time. For instance, now that I've found a way in which I prefer to format my matplotlib figures, I don't want to have to copy and paste the corresponding 10 lines of code every time. Instead, I prefer to define a callable function that implements this routine. This repo is full of such callable functions for tasks which I want to do often in many different projects.


---

# Demo: Making .gif's

[https://colab.research.google.com/drive/10sOSQChyJrrajtvnBRkfhUHAI-tNO6zQ?usp=sharing](https://colab.research.google.com/drive/10sOSQChyJrrajtvnBRkfhUHAI-tNO6zQ?usp=sharing)

---

# Prerequisites for Using This Code
This repo depends only on some standard libraries.

**List of Requirements in Order to Use this Code:**
- [x] Have python installed and know how to edit and run python files
- [x] For full functionality, have the prerequisite standard packages installed: `numpy`, `scipy`, `matplotlib`, `plotly`, `tensorflow`, `pytorch`, and `sklearn`. However, e.g., `tensorflow` is not necessary for `my_torch_utils` and `pytorch` is not necessary for `my_keras_utils`.

---

# Installation

Unfortunately, you need to install the dependencies manually (see **Prerequisites for Using This Code**), at least for now (see [#4](https://github.com/ThomasLastName/quality-of-life/issues/4)). Having the dependencies installed, try 
> `pip install --upgrade git+https://github.com/ThomasLastName/quality-of-life.git`

Failing that, try cloning this repo to wherever you want and then add the directory of the repo to the PATH for your environment (or system). I think this can be accomplished in 4 steps? First `cd wherever/the/hell`, then `git clone https://github.com/ThomasLastName/quality-of-life.git`, next `cd quality-of-life`, and finally `pip install -e .`?

---

# Documentation
Boy, I sure should write some!

---

# Usage
That's up to you! However, please credit me with a comment along the lines of `# ~~~ Tom wrote these; maintained at https://github.com/ThomasLastName/quality_of_life` in your code, if you use these!

---

## Recommend Application: Green Ouput, Yellow Warnings, and Red Errors
After installing this code (wchich basically consists of putting a folder called `quality_of_life` containing these `.py` files in your `Lib` folder), I recommend creating a `usercustomize.py` file in your `Lib` folder containing the following code
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

or, if you already have a `usercustomize.py` file, consider adding the above code to it. Additionally, in the source code for `warnings.py` (also in your `Lib` folder), I recommend modifying the definition of `_showwarnmsg_impl` as follows so that warnings will print in yellow

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
