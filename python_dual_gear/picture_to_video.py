"""
picture to video utility
"""
import logging
import cv2
import os
import sys
from copy import deepcopy

# how long is this video, in seconds
video_length_sec = 10
# root directory of models
models_root_dir = r'C:\Projects\gears\python_dual_gear\debug\pending'
# names not treated as a model
ignored_names = ['picture_to_video.py', '.DS_Store']
# folder to be searched inside each model
folder_name = 'png'
# file prefix of the photos
prefix = ''
# also attach a reversed version
attach_reverse = True
# replay
replay_times = 0

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
models_root_dir = os.path.abspath(models_root_dir)
models = os.listdir(models_root_dir)
models = [model for model in models if model not in ignored_names]

for model in models:
    folder = os.path.join(models_root_dir, model, folder_name)
    if not os.path.isdir(folder):
        continue

    # filter the files of interest
    image_files = [filename for filename in os.listdir(folder) if prefix in filename and '.png' in filename]
    image_files = [filename for filename in image_files if os.path.isfile(os.path.join(folder, filename))]
    image_files = [os.path.splitext(filename)[0][len(prefix):] for filename in image_files]
    image_files.sort(key=float)
    count = len(image_files)
    logger.info(f'{count} pictures retrieved from {model}')

    # the file name of this video
    video_name = f"{model}.mp4"

    img_array = []
    for image_file in image_files:
        filename = os.path.abspath(os.path.join(folder, prefix + image_file + '.png'))
        logger.info(f'Processed {filename}')
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    if attach_reverse:
        rev_img_array = deepcopy(img_array)[::-1]
        img_array = img_array + rev_img_array
        logger.info('reverse data attached')

    if replay_times > 0:
        unit_part = deepcopy(img_array)
        for i in range(replay_times):
            img_array += deepcopy(unit_part)

    out = cv2.VideoWriter(os.path.join(models_root_dir, video_name), cv2.VideoWriter_fourcc(*'mp4v'),
                          len(img_array) / video_length_sec, size)

    for img in img_array:
        out.write(img)
    out.release()

    print(f"video saved as {video_name}")
