import os
import sys
import bz2
import argparse
from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
import multiprocessing
import face_alignment
from skimage import io

import matplotlib.pyplot as plt

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

def get_align_images(raw_dir, aligned_dir, output_size=1024, x_scale=1, y_scale=1, em_scale=0.1, use_alpha=False):
    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))
    RAW_IMAGES_DIR = raw_dir
    ALIGNED_IMAGES_DIR = aligned_dir
    try:
        os.mkdir(ALIGNED_IMAGES_DIR)
    except:
        pass

    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for img_name in os.listdir(RAW_IMAGES_DIR):
#         print('Aligning %s ...' % img_name)
        try:
            raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
            fn = face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], 1)
            if os.path.isfile(fn):
                continue
#             print('Getting landmarks...')
            for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                try:
#                     print('Starting face alignment...')
                    face_img_name = '%s.png' % (os.path.splitext(img_name)[0])
                    aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
                    image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=output_size, x_scale=x_scale, y_scale=y_scale, em_scale=em_scale, alpha=use_alpha)
                    print('Wrote result %s' % aligned_face_path)
                except:
                    pass
#                     print("Exception in face alignment!")
        except:
            pass
#             print("Exception in landmark detection!")

def get_prediction_directory(path, device="cpu"):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    preds = fa.get_landmarks_from_directory(path)
    return preds
    
def get_prediction_file(path, device="cpu"):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    input = io.imread(path)
    preds = fa.get_landmarks(input)
    return preds

def get_draw_prediction_directory(path, device="cpu"):
    preds = get_prediction_directory(path)
    for img in preds:
        _, ax = plt.subplots(figsize=(5,5), dpi=80)
        im = plt.imread(img)
        ax.imshow(im)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.scatter(x=preds[img][0][:, 0], y=preds[img][0][:, 1], c='w', s=10)
    return preds

def get_draw_prediction_file(path, device="cpu"):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    input = io.imread(path)
    preds = fa.get_landmarks(input)
    plt.imshow(input)
    plt.yticks([])
    plt.xticks([])
    plt.scatter(x=preds[0][:, 0], y=preds[0][:, 1], c='w', s=10)
    return preds
