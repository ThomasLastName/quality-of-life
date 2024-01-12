
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/quality_of_life

import inspect
import warnings
import traceback
import sys
from contextlib import contextmanager
warnings.simplefilter("always",UserWarning)
from quality_of_life.ansi import bcolors


def my_warn(message,*args,**kwargs):
    frame = inspect.currentframe()
    try:
        caller_frame = frame.f_back
        file_name = caller_frame.f_code.co_filename
        line_number = caller_frame.f_lineno
        calling_function = caller_frame.f_code.co_name
        print(bcolors.WARNING)  # also introduces a line break, which can be avoided using the `write = ... ; writer.write(...)` syntax found in other functions in this module
        warnings.warn( "line " +
                      bcolors.HEADER + f"{line_number}" +
                      bcolors.WARNING + " of " +
                      bcolors.HEADER + f"{file_name}" +
                      bcolors.WARNING + " in " + 
                      bcolors.HEADER + f"{calling_function}:" +
                      bcolors.WARNING + f"{message}" + "\n" + bcolors.ENDC,
            UserWarning,
            stacklevel=3,
            *args,
            **kwargs
        )
    finally:
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
def support_for_progress_bars( warn=False, default_color=bcolors.OKGREEN ):
    # Before any of the code
    try:
        #~~~ Note the current setting and choose what to switch to
        old_color = revert_console(note_color=True)
        new_color = old_color if (old_color is not None) else default_color
        #~~~ Cancel whatever setting may have been active and, instead, switch to monochrome
        monochrome(new_color)
        #~~~ Do the actual code
        yield
    #~~~ After all the code
    finally:
        #~~~~ Reinstate the cancelled behavior
        if old_color is None:
            revert_console()    # out of the box settings
        else:
            colored_console_output( main_color=old_color, warn=warn )


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