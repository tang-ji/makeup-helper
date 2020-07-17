import os, requests, time, shutil
import bz2
import numpy as np
import matplotlib.pyplot as plt
import cv2

from keras.utils import get_file
from multiprocessing.dummy import Pool

from face_alignment.landmarks_detector import LandmarksDetector
from face_alignment.face_alignment import image_align


LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
azureKey="b2101b2ed9744c51b7dd0c0e7ecad979"



def search(search_term, azureKey):
    print('searching using bing: "'+search_term+'"')
    search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
    subscription_key = azureKey
    assert subscription_key
    headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
    params  = {"q": search_term, "imageType": "Photo","count":100}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    result=[]
    for i in search_results['value']:
        result.append(i['thumbnailUrl'])
    return result


def download(links_in, dir_name, n=None):
    links = links_in[:n]
    print('search results',len(links))
    if os.path.exists(dir_name):
        print('using cache')
        return

    tempName=dir_name+'-'+str(int(time.time()))
    os.makedirs(tempName)
    def fetch(url):
        r=requests.get(url[0], stream=True)
        with open(tempName+'/'+str(url[1]).zfill(4)+".jpg", 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)

    modLinks=[[links[i],i+1] for i in range(len(links))]
    Pool(10).map(fetch, modLinks)
    
    try:
        os.rename(tempName,dir_name)
    except:
        shutil.rmtree(tempName)

    print('Items downloaded',len(links))

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))
landmarks_detector = LandmarksDetector(landmarks_model_path)

def get_aligned_images(raw_dir, aligned_dir, output_size=1024, x_scale=1, y_scale=1, em_scale=0.1, use_alpha=False, verbose=True):
    RAW_IMAGES_DIR = raw_dir
    ALIGNED_IMAGES_DIR = aligned_dir
    try:
        os.mkdir(ALIGNED_IMAGES_DIR)
    except:
        pass

    for img_name in os.listdir(RAW_IMAGES_DIR):
        try:
            if img_name.split(".")[-1].lower() not in ["png", "jpg", "jpeg"]:
                continue
            raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
            fn = face_img_name = '%s.png' % (os.path.splitext(img_name)[0])
            for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                try:
                    face_img_name = '%s.png' % (os.path.splitext(img_name)[0])
                    aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
                    image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=output_size, x_scale=x_scale, y_scale=y_scale, em_scale=em_scale, alpha=use_alpha)
                    if verbose:
                        print('Wrote result %s' % aligned_face_path)
                except:
                    pass
        except:
            pass

def get_landmarks_img(raw_img_path):
    preds = {}
    detector = landmarks_detector.get_landmarks(raw_img_path)
    preds[raw_img_path] = np.array(next(detector))
    return preds

def get_landmarks_path(raw_dir):
    RAW_IMAGES_DIR = raw_dir
    preds = {}
    for img_name in os.listdir(RAW_IMAGES_DIR):
        try:
            if img_name.split(".")[-1].lower() not in ["png", "jpg", "jpeg"]:
                continue
            raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
            for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                preds[raw_img_path] = np.array(face_landmarks)
        except:
            pass
            
    return preds

def get_draw_prediction_directory(path):
    preds = get_landmarks_path(path)
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

def get_features(landmarks):
    predictions = []
    predictions.append(("left eyebrow", get_boxes(landmarks[17:22])))
    predictions.append(("right eyebrow", get_boxes(landmarks[22:27])))
    predictions.append(("nose", get_boxes(np.append(landmarks[27:31], [landmarks[39], landmarks[42]], axis=0))))
    predictions.append(("nostrils", get_boxes(landmarks[31:36])))
    predictions.append(("left eye", get_boxes(landmarks[36:42])))
    predictions.append(("right eye", get_boxes(landmarks[42:48])))
    predictions.append(("upper lip", get_boxes(landmarks[48:55])))
    predictions.append(("lower lip", get_boxes(np.append(landmarks[54:60], [landmarks[48]], axis=0))))
    predictions.append(("left jaw", get_boxes(np.append(landmarks[1:6], [landmarks[48]], axis=0))))
    predictions.append(("right jaw", get_boxes(np.append(landmarks[11:16], [landmarks[54]], axis=0))))
    predictions.append(("chin", get_boxes(landmarks[5:12])))
    return predictions

def drawImage(image, ax=None):
    """Draw image.

    Args:
        image: The image to draw
        ax: A matplotlib axis on which to draw.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8,8), dpi=80)
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
        _, ax = plt.subplots(figsize=(8,8), dpi=80)
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
