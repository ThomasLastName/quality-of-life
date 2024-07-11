
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/quality_of_life

import platform
import os
import sys

#
#~~~ Set up colored printing in the console
if platform.system()=="Windows":    # platform.system() returns the OS python is running o0n | see https://stackoverflow.com/q/1854/11595884
    os.system("color")              # Makes ANSI codes work | see Szabolcs' comment on https://stackoverflow.com/a/15170325/11595884
class bcolors:                      # https://stackoverflow.com/a/287944/11595884
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def header(message,*args,**kwargs):
    print( bcolors.HEADER, message, bcolors.ENDC, *args, **kwargs )

def blue(message,*args,**kwargs):
    print( bcolors.OKBLUE, message, bcolors.ENDC, *args, **kwargs )

def cyan(message,*args,**kwargs):
    print( bcolors.OKCYAN, message, bcolors.ENDC, *args, **kwargs )

def green(message,*args,**kwargs):
    print( bcolors.OKGREEN, message, bcolors.ENDC, *args, **kwargs )

def fail(message,*args,**kwargs):
    print( bcolors.FAIL, message, bcolors.ENDC, *args, **kwargs )

def bold(message):
    return  bcolors.BOLD + message + bcolors.ENDC

def underline(message):
    return  bcolors.UNDERLINE + message + bcolors.ENDC


"""
~
~ RECOMMENDED: in the source code for warnings.py, modify the definition of _showwarnmsg_impl as follows to achieve yellow warnings
~

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
"""