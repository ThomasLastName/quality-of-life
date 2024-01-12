# QAULITY OF LIFE HELPER ROUTINES
These are some helper routines that I want to be able to load without rewriting them every time. For instance, now that I've found a way in which I prefer to format my matplotlib figures, I don't want to have to copy and paste the corresponding 10 lines of code every time. Instead, I prefer to define a callable function that implements this routine. This repo is full of such callable functions for tasks which I want to do often in many different projects. That said, if you have multiple versions of python on you'll computer, then you may want to be mindful of which version's terminal you're executing `import os; import sys; print(os.path.dirname(sys.executable))` in.

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

---

# Prerequisites for Using This Code
This repo depends only on some standard libraries.

**List of Requirements in Order to Use this Code:**
- [x] Have python installed and know how to edit and run python files
- [x] **(important)** Know the directory of your python's `Lib` folder (see below)
- [x] Have the prerequisite standard packages installed: `numpy`, `matplotlib`, `tensorflow`, `pytorch`, and `sklearn`.

**More on the Directory of Your Python's `Lib` Folder:** Unless you made a point of moving python after installing it, this will be the directory to which you installed python, plus `\Lib`. For example, on my personal computer, python is located in the folder  `C:\Users\thoma\AppData\Local\Programs\Python\Python310`, within which many things are contained, including a folder called `Lib`. Thus, the directory of my `Lib` folder is `C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib`. For reference, this is also where many of python's base modules are stored, such as `warnings.py`, `pickle.py`, and `turtle.py`.

I recommend having the directory where your version python is installed written down somewhere. If you do not know this location, I believe you can retrieve it in the interactive python terminal by commanding `import os; import sys; print(os.path.dirname(sys.executable))`. Thus, in Windows, you can probably just open the command line and paste into it `python -c "import os; import sys; print(os.path.dirname(sys.executable))"`. 

---

# Installation

Basically, just create a folder called `quality_of_life` inside of your python's `Lib` folder, and fill it with the files from this repository.

---

## Detailed Installation Instructions Using git (recommended)

**Additional Prerequisites Using git:**
- [x] Have git installed on your computer

**Installation Steps Using git:**
Navigate  to the `Lib` folder of the version of python you want to use. Once there, command `git clone https://github.com/ThomasLastName/quality_of_life.git`, which will create and populate a folder called `quality_of_life` in the same directory.

For example, given that the directory of my `Lib` folder is `C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib` on my personal computer, I would navigate there by pasting `cd C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib` into the Windows command line, and then I would paste `git clone https://github.com/ThomasLastName/quality_of_life.git`.

**Subsequent Updates Using git:**
Navigate to the directory of the folder that you created, and within that directory command `git pull https://github.com/ThomasLastName/quality_of_life.git`.

For instance, to continue the example above, if I created the folder `quality_of_life` in my `Lib` folder `C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib`, then the directory of the folder `quality_of_life` is `C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib\quality_of_life`. I'll want to navigate there in the Windows command line by pasting `cd C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib\quality_of_life` and, then, I'm ready to paste `git pull https://github.com/ThomasLastName/quality_of_life.git`.

---

## Detailed Installation Instructions Using the Graphical Interface

**Installation Steps Using the Graphical Interface:**
Click the colorful `<> Code` button at [https://github.com/ThomasLastName/quality_of_life](https://github.com/ThomasLastName/quality_of_life) and select `Download ZIP` from the dropdown menu. This should download a zipped folder called `quality_of_life` containing within it an unzipped folder of the same name, which you just need to click and drag (or copy and paste) into the `Lib` folder of your preferred version of python.

**Subsequent Updates Using the Graphical Interface:**
You'll have to repeat the process, again. When you attempt to click and drag (or copy and paste) the next time, your operating system probably prompts you with something like "These files already exist! Are you tweaking or did you want to replace them?" and you can just click "replace" or whatever it prompts you with.

