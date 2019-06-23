

class Model():

    def __init__(self, name, sample_num, center_point, tooth_height, tooth_num, k=1 ):
        self.name = name
        self.sample_num = sample_num
        self.center_point = center_point
        self.tooth_height = tooth_height
        self.tooth_num = tooth_num
        self.k = k



our_models = [Model(name = 'mahou2',               sample_num=512,  center_point = (390, 229), tooth_height= 10, tooth_num= 3    ),
              Model(name = 'mahou',                sample_num=512,  center_point = (710, 437), tooth_height= 15, tooth_num= 32   ),
              Model(name = 'wolf',                 sample_num=512,  center_point = (300, 300), tooth_height= 10,  tooth_num= 128  ),
              Model(name = 'irregular_circle',     sample_num=512,  center_point = (480, 214), tooth_height= 8,  tooth_num= 32   ),
              Model(name = 'ellipse',              sample_num=32,   center_point = (438, 204), tooth_height= 8,  tooth_num= 32   ),
              Model(name = 'spiral_circle_convex', sample_num=512,  center_point = (470, 206), tooth_height= 8,  tooth_num= 32   ),
              Model(name = 'man',                  sample_num=4096, center_point = (93, 180),  tooth_height= 1,  tooth_num= 128  )]