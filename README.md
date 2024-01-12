# QAULITY OF LIFE HELPER ROUTINES
These are some helper routines that I want to be able to load without rewriting them every time. For instance, now that I've found a way in which I prefer to format my matplotlib figures, I don't want to have to copy and paste the corresponding 10 lines of code every time. Instead, I prefer to define a callable function that implements this routine. This repo is full of such callable functions for tasks which I want to do often in many different projects.

---

# Usage
That's up to you! However, please credit me with a comment like `# from https://github.com/ThomasLastName/quality_of_life` in your code, if you use these!

---

# Prerequisites for Using This Code
This repo depends on some standard libraries.

**List of Requirements in Order to Use this Code:**
- [x] Have python installed and know how to edit and run python files
- [x] **(important)** Know the directory of your python's `Lib` folder (see below)
- [x] Have the prerequisite standard packages installed: `numpy`, `matplotlib`, and `quality_of_life` `tensorflow`, `pytorch`, `sklearn`, and `alive_progress`

**More on the Directory of Your Python's `Lib` Folder:** Unless you made a point of moving python after installing it, this will be the directory to which you installed python, plus `\Lib`. For example, on my personal computer, python is located in the folder  `C:\Users\thoma\AppData\Local\Programs\Python\Python310`, within which many things are contained, including a folder called `Lib`. Thus, the directory of my `Lib` folder is `C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib`. For reference, this is also where many of python's base modules are stored, such as `warnings.py`, `pickle.py`, and `turtle.py`.

I recommend having the directory where your version python is installed written down somewhere. If you do not know this location, I believe you can retrieve it in the interactive python terminal by commanding `import os; import sys; print(os.path.dirname(sys.executable))`. Thus, in Windows, you can probably just open the command line and paste into it `python -c "import os; import sys; print(os.path.dirname(sys.executable))"`. 

---

# Installation Using git (recommended)

**Additional Prerequisites Using git:**
- [x] Have git installed on your computer

**Installation Steps Using git:**
Navigate  to the `Lib` folder of the version of python you want to use. Once there, command `git clone https://github.com/ThomasLastName/quality_of_life.git`, which will create and populate a folder called `quality_of_life` in the same directory.

For example, given that the directory of my `Lib` folder is `C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib` on my personal computer, I would navigate there by pasting `cd C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib` into the Windows command line, and then I would paste `git clone https://github.com/ThomasLastName/quality_of_life.git`.

**Subsequent Updates Using git:**
Navigate to the directory of the folder that you created, and within that directory command `git pull https://github.com/ThomasLastName/quality_of_life.git`.

For instance, to continue the example above, if I created the folder `quality_of_life` in my `Lib` folder `C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib`, then the directory of the folder `quality_of_life` is `C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib\quality_of_life`. I'll want to navigate there in the Windows command line by pasting `cd C:\Users\thoma\OneDrive\Desktop\quality_of_life` and, then, I'm ready to paste `git pull https://github.com/ThomasLastName/quality_of_life.git`.

---

# Installation Using the Graphical Interface

**Installation Steps Using the Graphical Interface:**
Click the colorful `<> Code` button at [https://github.com/ThomasLastName/quality_of_life](https://github.com/ThomasLastName/quality_of_life) and select `Download ZIP` from the dropdown menu. This should download a zipped folder called `quality_of_life` containing within it an unzipped folder of the same name, which you just need to click and drag (or copy and paste) into the `Lib` folder of your preferred version of python.

**Subsequent Updates Using the Graphical Interface:**
You'll have to repeat the process, again. When you attempt to click and drag (or copy and paste) the next time, your operating system probably prompts you with something like "These files already exist! Are you tweaking or did you want to replace them?" and you can just click "replace" or whatever it prompts you with.

