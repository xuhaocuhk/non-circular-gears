class Model():

    def __init__(self, name, sample_num, center_point, tooth_height, tooth_num, k=1):
        self.name = name
        self.sample_num = sample_num
        self.center_point = center_point
        self.tooth_height = tooth_height
        self.tooth_num = tooth_num
        self.k = k


our_models = [Model(name='circular', sample_num=32, center_point=(0, 0), tooth_height=0.03, tooth_num=32),
              Model(name='ellipse', sample_num=1024, center_point=(0, 0), tooth_height=0.03, tooth_num=32),
              Model(name='focal_ellipse', sample_num=512, center_point=(0,0), tooth_height=0.03, tooth_num=32),
              Model(name='irregular_circle', sample_num=512, center_point=(0, 0), tooth_height=0.03, tooth_num=32),
              Model(name='irregular_ellipse', sample_num=1024, center_point=(0, 0), tooth_height=0.03, tooth_num=32),
              Model(name='spiral_circle', sample_num=512, center_point=(0, 0), tooth_height=0.03, tooth_num=32),
              Model(name='mahou', sample_num=1024, center_point=(0.0, 0.0), tooth_height=0.02, tooth_num=32),
              Model(name='wolf', sample_num=4096, center_point=(0, 0), tooth_height=0.03, tooth_num=128),
              Model(name='trump', sample_num=1024, center_point=(0, 0), tooth_height=0.03, tooth_num=64),
              Model(name='man', sample_num=4096, center_point=(0, 0), tooth_height=0.03, tooth_num=32)]
