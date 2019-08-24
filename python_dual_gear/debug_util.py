import time
import datetime
import os
import logging
from typing import Union, Iterable, Callable
from multiprocessing import Process
import sys


class MyDebugger:
    pre_fix = 'debug'

    def __init__(self, model_name: Union[str, Iterable[str]]):
        if isinstance(model_name, str):
            self.model_name = model_name
        else:
            self.model_name = '_'.join(model_name)
        self._debug_dir_name = os.path.join(MyDebugger.pre_fix, datetime.datetime.fromtimestamp(time.time()).strftime(
            f'%Y-%m-%d_%H-%M-%S_{self.model_name}'))
        self._debug_dir_name = os.path.join(os.path.dirname(__file__), self._debug_dir_name)
        logging.info("=================== Program Start ====================")
        logging.info(f"Output directory: {self._debug_dir_name}")
        self._init_debug_dir()

    def file_path(self, file_name):
        return os.path.join(self._debug_dir_name, file_name)

    def get_root_debug_dir_name(self):
        return self._debug_dir_name

    def get_math_debug_dir_name(self):
        return os.path.join(self._debug_dir_name, "math_rotate")

    def get_cutting_debug_dir_name(self):
        return os.path.join(self._debug_dir_name, "cut_rotate")

    def _init_debug_dir(self):
        # init root debug dir
        if not os.path.exists(MyDebugger.pre_fix):
            os.mkdir(MyDebugger.pre_fix)
        os.mkdir(self._debug_dir_name)
        logging.info("Directory %s established" % self._debug_dir_name)
        os.mkdir(os.path.join(self._debug_dir_name, "math_rotate"))
        logging.info("Directory %s established" % os.path.join(self._debug_dir_name, "math_rotate"))
        os.mkdir(os.path.join(self._debug_dir_name, "cut_rotate"))
        logging.info("Directory %s established" % os.path.join(self._debug_dir_name, "cut_rotate"))


class SubprocessDebugger:
    """
    the debugger that starts a subprocess for a given function
    """

    def __init__(self, debugger: MyDebugger, function: Callable, args: tuple = ()):
        self.debugger = debugger
        self.function = function
        self.args = args
        self.process = None

    def _func(self):
        sys.stdout = open(self.debugger.file_path('stdout.txt'), 'w')
        sys.stderr = open(self.debugger.file_path('stderr.txt'), 'w')
        self.function(*self.args)

    def start(self):
        if self.process is not None:
            raise RuntimeError("SubprocessDebugger Not Restartable")
        self.process = Process(target=self._func)
        self.process.start()

    def join(self):
        """
        wait the debugging process to finish
        """
        if self.process is None:
            raise RuntimeError("Subprocess Not Started")
        self.process.join()


def __main__test__function(args):
    """
    function merely meant to test the debugger used by '__name__==__main__' part, do not import
    """
    print('this shall be stdout' + repr(args))
    error = 1 / 0


if __name__ == '__main__':
    p_debugger = SubprocessDebugger(MyDebugger('hello'), __main__test__function, ('233',))
    p_debugger.start()
    p_debugger.join()
