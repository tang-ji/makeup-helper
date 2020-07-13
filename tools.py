import os
import bz2
import numpy as np
from keras.utils import get_file
from landmarks_detector import LandmarksDetector

import matplotlib.pyplot as plt

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))

def get_aligned_images(raw_dir, aligned_dir, output_size=1024, x_scale=1, y_scale=1, em_scale=0.1, use_alpha=False):
    RAW_IMAGES_DIR = raw_dir
    ALIGNED_IMAGES_DIR = aligned_dir
    try:
        os.mkdir(ALIGNED_IMAGES_DIR)
    except:
        pass

    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for img_name in os.listdir(RAW_IMAGES_DIR):
        try:
            raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
            fn = face_img_name = '%s.png' % (os.path.splitext(img_name)[0])
            for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                try:
                    face_img_name = '%s.png' % (os.path.splitext(img_name)[0])
                    aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
                    image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=output_size, x_scale=x_scale, y_scale=y_scale, em_scale=em_scale, alpha=use_alpha)
                    print('Wrote result %s' % aligned_face_path)
                except:
                    pass
        except:
            pass

def get_landmarks(raw_dir):
    RAW_IMAGES_DIR = raw_dir
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    preds = {}
    for img_name in os.listdir(RAW_IMAGES_DIR):
        try:
            raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
            for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                preds[raw_img_path] = np.array(face_landmarks)
        except:
            pass
            
    return preds

def get_draw_prediction_directory(path):
    preds = get_landmarks(path)
    for img in preds:
        _, ax = plt.subplots(figsize=(5,5), dpi=80)
        im = plt.imread(img)
        ax.imshow(im)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.scatter(x=preds[img][:, 0], y=preds[img][:, 1], c='w', s=10)
    return preds
