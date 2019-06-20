import gzip
import math
import os
import os.path
import pickle
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from scipy.ndimage import distance_transform_edt
import sys

def ensuredir(dirname):
    """Ensure a directory exists"""
    if not os.path.exists(dirname):
            os.makedirs(dirname)

'''
Turn a number into a string that is zero-padded up to length n
'''
def zeropad(num, n):
    sn = str(num)
    while len(sn) < n:
            sn = '0' + sn
    return sn

def pickle_dump_compressed(object, filename, protocol=pickle.HIGHEST_PROTOCOL):
    """Pickles + compresses an object to file"""
    file = gzip.GzipFile(filename, 'wb')
    file.write(pickle.dumps(object, protocol))
    file.close()

def pickle_load_compressed(filename):
    """Loads a compressed pickle file and returns reconstituted object"""
    file = gzip.GzipFile(filename, 'rb')
    buffer = b""
    while True:
            data = file.read()
            if data == b"":
                    break
            buffer += data
    object = pickle.loads(buffer)
    file.close()
    return object
        
def get_data_root_dir():
    """
    Get root dir of the data, defaults to /data if env viariable is not set
    """
    env_path = os.environ.get("SCENESYNTH_DATA_PATH")
    if env_path:
        return env_path
    else:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        return f"{root_dir}/data"

def memoize(func):
    """
    Decorator to memoize a function
    https://medium.com/@nkhaja/memoization-and-decorators-with-python-32f607439f84
    """
    cache = func.cache = {}

    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key not in cache:
                    cache[key] = func(*args, **kwargs)
            return cache[key]

    return memoized_func

@contextmanager
def stdout_redirected(to=os.devnull):
    """
    From https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
    Suppress C warnings
    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different
