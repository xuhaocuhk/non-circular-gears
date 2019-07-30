"""
Functions used for testing
"""
from typing import List, Tuple, Dict, Any, Union, Iterable
import numpy as np
from debug_util import MyDebugger, SubprocessDebugger
from core.optimize_dual_shapes import sampling_optimization
import yaml


def optimize_pair_from_config(drive_contour: np.ndarray, driven_contour: np.ndarray, debugger: MyDebugger,
                              configuration: Union[str, Dict[str, Any]]):
    """
    optimize a pair with the optimization config given
    :param drive_contour: the drive contour to be optimized to
    :param driven_contour: the driven contour to be optimized to
    :param debugger: the debugger to provide the path for storing related files
    :param configuration: optimization config, either as yaml filename or a dictionary. See the yaml file for details.
    :return:
    """
    if isinstance(configuration, str):
        with open(configuration) as yaml_file:
            configuration = yaml.safe_load(yaml_file)
            configuration['sampling_count'] = tuple(configuration['sampling_count'])
    sampling_optimization(drive_contour, driven_contour, debugger=debugger, visualization={}, draw_tar_functions=True,
                          **configuration)


def optimization_test(names: List[Iterable[str]], optimize_pairs: List[Tuple[np.ndarray, np.ndarray]], config_file: str,
                      parallel: bool = False):
    """
    test optimization for multiple pairs in sub-processes
    :param names: names for each pair for debug use
    :param optimize_pairs: the pairs to be optimized to
    :param config_file: configuration file for optimization
    :param parallel: run subprocesses in parallel
    :return:
    """
    processes = []
    for name, pair in zip(names, optimize_pairs):
        debugger = MyDebugger(name)
        process = SubprocessDebugger(debugger, optimize_pair_from_config, (*pair, debugger, config_file))
        process.start()
        if parallel:
            processes.append(process)
        else:
            process.join()
    for process in processes:
        process.join()


if __name__ == '__main__':
    from shape_factory import get_shape_contour
    from models import our_models

    models = {
        model.name: get_shape_contour(model, True, None, model.smooth) for model in our_models
    }
    test_case_names = [
        ('australia', 'square'),
        ('united_states', 'square'),
        ('france', 'square'),
    ]

    optimization_test(test_case_names,
                      [(models[drive_name], models[dual_name]) for drive_name, dual_name in test_case_names],
                      'optimization_config.yaml',
                      False)
