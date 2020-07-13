import os
import bz2
import numpy as np
from keras.utils import get_file
from landmarks_detector import LandmarksDetector

import matplotlib.pyplot as plt
import cv2

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

def get_bound(points):
    x_max = np.max(points, axis=0)[0]
    x_min = np.min(points, axis=0)[0]
    y_max = np.max(points, axis=0)[1]
    y_min = np.min(points, axis=0)[1]
    return x_max, x_min, y_max, y_min

def get_boxes(points):
    x_max, x_min, y_max, y_min = get_bound(points)
    return np.array([[x_min,y_min], [x_max,y_min], [x_max,y_max], [x_min, y_max]])

def drawImage(image, ax=None):
    """Draw image.

    Args:
        image: The image to draw
        ax: A matplotlib axis on which to draw.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6,6), dpi=100)
    ax.imshow(image)
    ax.set_yticks([])
    ax.set_xticks([])
    
def drawBoxes(image, boxes, color=(255, 0, 0), thickness=5, boxes_format='boxes'):
    """Draw boxes onto an image.

    Args:
        image: The image on which to draw the boxes.
        boxes: The boxes to draw.
        color: The color for each box.
        thickness: The thickness for each box.
        boxes_format: The format used for providing the boxes. Options are
            "boxes" which indicates an array with shape(N, 4, 2) where N is the
            number of boxes and each box is a list of four points) as provided
            by `keras_ocr.detection.Detector.detect`, "lines" (a list of
            lines where each line itself is a list of (box, character) tuples) as
            provided by `keras_ocr.data_generation.get_image_generator`,
            or "predictions" where boxes is by itself a list of (word, box) tuples
            as provided by `keras_ocr.pipeline.Pipeline.recognize` or
            `keras_ocr.recognition.Recognizer.recognize_from_boxes`.
    """
    if len(boxes) == 0:
        return image
    canvas = image.copy()
    if boxes_format == 'lines':
        revised_boxes = []
        for line in boxes:
            for box, _ in line:
                revised_boxes.append(box)
        boxes = revised_boxes
    if boxes_format == 'predictions':
        revised_boxes = []
        for _, box in boxes:
            revised_boxes.append(box)
        boxes = revised_boxes
    for box in boxes:
        cv2.polylines(img=canvas,
                      pts=box[np.newaxis].astype('int32'),
                      color=color,
                      thickness=thickness,
                      isClosed=True)
    return canvas

def drawAnnotations(image, predictions, ax=None):
    """Draw text annotations onto image.
    Args:
        image: The image on which to draw
        predictions: The predictions as provided by `pipeline.recognize`.
        ax: A matplotlib axis on which to draw.
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(drawBoxes(image=image, boxes=predictions, boxes_format='predictions'))
    predictions = sorted(predictions, key=lambda p: p[1][:, 1].min())
    left = []
    right = []
    for word, box in predictions:
        if box[:, 0].min() < image.shape[1] / 2:
            left.append((word, box))
        else:
            right.append((word, box))
    ax.set_yticks([])
    ax.set_xticks([])
    for side, group in zip(['left', 'right'], [left, right]):
        for index, (text, box) in enumerate(group):
            y = 1 - (index / len(group))
            xy = box[0] / np.array([image.shape[1], image.shape[0]])
            xy[1] = 1 - xy[1]
            ax.annotate(s=text,
                        xy=xy,
                        xytext=(-0.05 if side == 'left' else 1.05, y),
                        xycoords='axes fraction',
                        arrowprops={
                            'arrowstyle': '-',
                            'color': 'r'
                        },
                        color='r',
                        fontsize=14,
                        horizontalalignment='right' if side == 'left' else 'left')
    return ax
