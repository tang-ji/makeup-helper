import face_alignment
from skimage import io

import matplotlib.pyplot as plt

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
