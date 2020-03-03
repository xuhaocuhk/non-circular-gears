from typing import Tuple, List, Union
import yaml
import os


class Model:
    def __init__(self, name, sample_num, center_point, tooth_height, tooth_num, k=1, smooth=0):
        self.name = name
        self.sample_num = sample_num
        self.center_point = center_point
        self.tooth_height = tooth_height
        self.tooth_num = tooth_num
        self.k = k
        self.smooth = smooth


def load_model_from_file(data: dict) -> Model:
    data['center_point'] = tuple(data['center_point'])
    return Model(**data)


def load_models(filename: str) -> List[Model]:
    with open(filename) as file:
        return [load_model_from_file(data) for data in yaml.safe_load(file)]


our_models = load_models(os.path.join(os.path.dirname(__file__), 'models.yaml'))


def find_model_by_name(model_name: str) -> Union[Model, None]:
    for model in our_models:
        if model.name == model_name:
            return model
    return None


def generate_model_pool(model_names: Tuple[str]):
    model_pool = []
    for available_model in our_models:
        if available_model.name in model_names:
            model_pool.append(available_model)
    assert len(model_pool) == len(model_names)
    return model_pool


if __name__ == '__main__':
    print(our_models)
