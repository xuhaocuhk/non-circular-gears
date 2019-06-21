from debug_util import MyDebugger
from models import our_models
from examples import pick_center_point, cut_gear


if __name__ == '__main__':
    chosen_model = our_models[5]
    debugger = MyDebugger(chosen_model.name)
    # if models[chosen][2] is None:
    pick_center_point(chosen_model, debugger)
    cut_gear(chosen_model, debugger)
