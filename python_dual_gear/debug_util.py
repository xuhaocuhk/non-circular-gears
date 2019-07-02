import time
import datetime
import os
import logging

logging.basicConfig(level=logging.INFO)


class MyDebugger():
    pre_fix = 'debug'

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._debug_dir_name = os.path.join(MyDebugger.pre_fix, datetime.datetime.fromtimestamp(time.time()).strftime(
            f'%Y-%m-%d_%H-%M-%S_{model_name}'))
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


if __name__ == '__main__':
    # debugger = MyDebugger("xuhao")
    print(debugger.get_root_debug_dir_name())
