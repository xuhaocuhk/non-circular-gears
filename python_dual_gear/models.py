

class Model():

    def __init__(self, name, sample_num, center_point, tooth_height, tooth_num ):
        self.name = name
        self.sample_num = sample_num
        self.center_point = center_point
        self.tooth_height = tooth_height
        self.tooth_num = tooth_num



our_models = [Model('mahou2', 512, (390, 229), 10, 32),
              Model('mahou', 512, (710, 437), 15, 32),
              Model('wolf', 512, (300, 300), 5, 128),
              Model('irregular_circle', 512, (480, 214), 8, 32),
              Model('ellipse', 32, (438, 204), 8, 32),
              Model('spiral_circle_convex', 512, (470, 206), 8, 32),
              Model('man', 4096, (93, 180), 1, 128)]