
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/quality_of_life

import inspect
import warnings
import traceback
import sys
import os
import json
from contextlib import contextmanager
warnings.simplefilter("always",UserWarning)
from quality_of_life.ansi import bcolors

#
# ~~~ Save a dictionary as a .json; from https://stackoverflow.com/a/7100202/11595884
def dict_to_json( dict, path_including_file_extension, override=False, verbose=True ):
    #
    # ~~~ Check that the path is available
    not_empty = os.path.exists(path_including_file_extension)
    #
    # ~~~ If that path already exists and the user did not say "over-ride" it, then raise an error
    if not_empty and not override:
        raise ValueError("The specified path already exists. Operation halted. Specify `override=True` to override this halting.")
    #
    # ~~~ If the file path is either available, or the user gave permission to over-ride it, then proceed to write there
    with open(path_including_file_extension,'w') as fp:
        json.dump(dict,fp)
    #
    # ~~~ Print helpful messages
    if verbose:
        if override:
            my_warn(f"The path {path_including_file_extension} was not empty. It has been overwritten.")
        print(f"Created {path_including_file_extension} at {os.path.abspath(path_including_file_extension)}")

#
# ~~~ Load a .json as a dictionary; from https://stackoverflow.com/a/7100202/11595884
def json_to_dict(path_including_file_extension):
    with open(path_including_file_extension,'r') as fp:
        dict = json.load(fp)
    return dict

#
# ~~~ Return the truth value of the statement that `dict` is identical to `other_dict`
def dicts_are_identical( dict, other_dict ):
    bool = (dict.keys()==other_dict.keys())
    for key in dict:
        bool = min( bool, dict[key]==other_dict[key] )
    return bool

def clear_last_line(prompt):
    # Move cursor to the beginning of the line
    sys.stdout.write('\r')
    # Clear the line by printing spaces
    sys.stdout.write(' ' * (len(prompt)))
    # Move cursor back to the beginning of the line
    sys.stdout.write('\r')
    sys.stdout.flush()


def get_input_with_clear(prompt="Enter something: "):
    input_text = input(prompt)
    clear_last_line(prompt)
    return input_text

def load_txt(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def get_file_extension(file_path):
    return os.path.splitext(file_path)[1]


def get_file_name(file_path):
    return os.path.basename(file_path)


def modify_path(file_path_or_name,force=False):
    #
    # ~~~ If the path doesn't exist, then no modification is needed; do nothing
    if (not os.path.exists(file_path_or_name)) and (not force):
        return file_path_or_name
    #
    # ~~~ Remove any the doodads surring the file name, thus leaving the only thing we wish to modify
    original_extension = get_file_extension(file_path_or_name)
    file_name_and_extnesion = get_file_name(file_path_or_name)
    name_only = file_name_and_extnesion.replace(original_extension,"")
    #
    # ~~~ Check if the name ends with " (anything)"
    start = name_only.rfind("(")
    end = name_only.rfind(")")
    correct_format = name_only.endswith(")") and (not start==-1) #and name_only[start-1]==" "   # ~~~ note: .rfind( "(" ) returns -1 if "(" is not found
    #
    # ~~~ If the file name is like "text (2)", turn that into "text (3)"
    if correct_format:
        if correct_format:
            thing_inside_the_parentheses = name_only[start + 1:end]
            try:
                num = int(thing_inside_the_parentheses)
                new_num = num + 1
                modified_name = name_only[:start + 1] + str(new_num) + name_only[end:]
            except ValueError:
                #
                # ~~~ If conversion to int fails, treat it as if the name didn't end with a valid " (n)"
                correct_format = False
    #
    # ~~~ If the file name didn't end with " (n)" for some n, then just append " (1)" to the file name
    if not correct_format:
        modified_name = name_only + " (1)"
    #
    # ~~~ Reattach any doodads we removed
    return file_path_or_name.replace( file_name_and_extnesion, modified_name+original_extension )



def process_for_saving(file_path_or_name):
    while os.path.exists(file_path_or_name):
        file_path_or_name = modify_path(file_path_or_name)
    return file_path_or_name

"""
# Example usage:
name = "/path/to/file.txt"
modified_path = modify_path(name,force=True)
print(f"Original path: {name}")
print(f"Modified path: {modified_path}")
"""


def my_warn( message, format_of_message=bcolors.OKBLUE, *args, **kwargs ):
    frame = inspect.currentframe()
    try:
        caller_frame = frame.f_back
        file_name = caller_frame.f_code.co_filename
        line_number = caller_frame.f_lineno
        calling_function = caller_frame.f_code.co_name
        old_color = revert_console(note_color=True)
        with support_for_progress_bars():
            print(bcolors.WARNING)  # also introduces a line break, which can be avoided using the `write = ... ; writer.write(...)` syntax found in other functions in this module
            warnings.warn(
                        "line " +
                        bcolors.HEADER + f"{line_number}" +
                        bcolors.WARNING + " of " +
                        bcolors.HEADER + f"{file_name}" +
                        bcolors.WARNING + " in " + 
                        bcolors.HEADER + f"{calling_function}: " +
                        bcolors.WARNING + f"'{message}'" + "\n" +
                        format_of_message,
                    UserWarning,
                    stacklevel=3,
                    *args,
                    **kwargs
                )
    finally:
        if old_color is None:
            revert_console()    # ~~~ out of the box settings
        else:
            colored_console_output( main_color=old_color, warn=False )
        del frame

#
# ~~~ An alternative to the defauly sys.excepthook which, additionally, colors the error output in red
def red_excepthook(type, value, traceback_obj):
    # Get the original error message
    error_message = ''.join(traceback.format_exception(type, value, traceback_obj))
    # Append frowning face emoji to the original error message
    modified_error_message = bcolors.FAIL + error_message.rstrip() + bcolors.ENDC + " \U0001F62D"
    # Print the modified error message
    print( modified_error_message, file=sys.stderr )

#
#~~~ Override the default excepthook with the custom_excepthook
def red_errors():
    sys.excepthook = red_excepthook     # ~~~ not reversible without resatring the python terminal


#
# ~~~ Chat gpt was mostly responsible for this one. You'll need to ask it how this works
class ColorizedStdout:
    def __init__( self, current_stdout, main_color ):
        self.pending_newline = False  # Flag to track if a newline is pending
        self.original_stdout = sys.stdout if not hasattr( current_stdout, 'original_stdout' ) else current_stdout.original_stdout
        self.main_color = main_color
    def write(self, message):
        # Apply color formatting to the message before writing
        colored_message = f"{self.main_color}{message.strip()}{bcolors.ENDC}"
        if self.pending_newline:
            self.original_stdout.write('\n')  # Add a newline before subsequent lines
        self.original_stdout.write(colored_message)
        if message.endswith('\n'):
            self.pending_newline = False
        else:
            self.pending_newline = True
    def __getattr__(self, attr):
        # Pass other attribute calls to the original sys.stdout
        return getattr(self.original_stdout, attr)


#
#~~~ Reset button for what we're about to write
def revert_console(note_color=False):
    #
    #~~~ Task 1 of 2: stop any weird colors
    writer = sys.stdout.original_stdout if hasattr(sys.stdout,'original_stdout') else sys.stdout
    writer.write(bcolors.ENDC)
    #
    #~~~ Task 2 of 2: reinstate the original stdout, if applicable
    if hasattr( sys.stdout, 'original_stdout' ):
        color = sys.stdout.main_color
        sys.stdout = sys.stdout.original_stdout
        if note_color:
            return color


#
#~~~ Instate a new sys.stdout
def colored_console_output( main_color=bcolors.OKGREEN, warn=True, clean_slate=True ):
    if clean_slate:
        revert_console()
    if warn:
        my_warn("`colored_console_output` is not necessarily integrated with other packages that also modify `sys.stdout`. For example, `tqdm` normally modifies `sys.stdout` in order to display the progress bar. For package integration, indent the block corresponding to the progress bar, and preface it by the context line `with support_for_progress_bars():`, e.g.,\n\nfrom tqdm import trange\nwith support_for_progress_bars():\n    for j in trange(20):\n        dummy_var = j   # replace with your code")
    sys.stdout = ColorizedStdout( sys.stdout, main_color=main_color )


#
#~~~ After this is called, evenything -- both input and output -- gets printed in monochrome
def monochrome(color):
    writer = sys.stdout.original_stdout if hasattr(sys.stdout,'original_stdout') else sys.stdout
    writer.write(color)


#
# ~~~ Integrate with packages like tqdm and alive_progress
@contextmanager
def support_for_progress_bars( warn=False, color=bcolors.OKGREEN ):
    #
    # ~~~ Before any of the code
    try:
        #
        # ~~~ Note the current setting and choose what to switch to
        old_color = revert_console(note_color=True)
        new_color = old_color if (old_color is not None) else color
        #
        # ~~~ Cancel whatever setting may have been active and, instead, switch to monochrome
        monochrome(new_color)
        #
        # ~~~ Do the actual code
        yield
    #
    # ~~~ After all the code
    finally:
        #
        # ~~~ Reinstate the cancelled behavior
        if old_color is None:
            revert_console()    # ~~~ out of the box settings
        else:
            colored_console_output( main_color=old_color, warn=warn )

#
#~~~ Compute [min-c,max+c] where c>0 is a buffer
def buffer(vector,multiplier=0.05):
    a = min(vector)
    b = max(vector)
    extra = (b-a)*multiplier
    return [a-extra, b+extra]

#
# ~~~ Start from the current working directory
def peel_back_cwd(stopping_lambda):
    path = os.getcwd()
    while not stopping_lambda(path):
        path, dirname = os.path.split(path)
        if len(dirname)==0:
            raise OSError(f"The stopping criterion was not met by any ancestor of the working directory {os.getcwd()}")
    return path

#
# ~~~ Find the root directory of a git repository when run from anywhere in the repo
def find_root_dir_of_repo():
    contains_dot_git = lambda path: os.path.exists(os.path.join(path,".git"))
    return peel_back_cwd( contains_dot_git )

#
# ~~~ Format a dictionary for printing; from https://www.geeksforgeeks.org/python-pretty-print-a-dictionary-with-dictionary-value/
format_dict = lambda dict: json.dumps(dict,indent=4)

#
# ~~~ Pretty print a dictionary; from https://www.geeksforgeeks.org/python-pretty-print-a-dictionary-with-dictionary-value/
print_dict = lambda dict: print(format_dict(dict))


"""
~
~ RECOMMENDED: create a usercustomize.py file in Lib if none exists; once one exists, add the following to it
~

#
#~~~ Green outputs
from quality_of_life.my_base_utils import colored_console_output
colored_console_output(warn=False)

#
#~~~ Red errors
from quality_of_life.my_base_utils import red_errors
red_errors()

"""



"""
~
~ EXAMPLE USAGE
~
from quality_of_life.my_base_utils import colored_console_output   # if running in a new terminal
from quality_of_life.ansi import bcolors
import sys

colored_console_output()
print("This will be printed in green.")
print("Another green message.")
sys.stdout.original_stdout  # returns the original sys.stdout <_io.TextIOWrapper name='<stdout>' mode='w' encoding='utf-8'>
sys.stdout                  # returns <quality_of_life.my_base_utils.ColorizedStdout object at 0x0000021A61B91B70>

colored_console_output( bcolors.OKBLUE, warn=False )
print("This will be printed in blue.")
print("Another blue message.")
sys.stdout.original_stdout  # returns the original sys.stdout <_io.TextIOWrapper name='<stdout>' mode='w' encoding='utf-8'>
sys.stdout                  # returns <quality_of_life.my_base_utils.ColorizedStdout object at 0x0000021A61B91B70>
"""