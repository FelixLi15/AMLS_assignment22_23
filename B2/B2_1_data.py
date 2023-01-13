import os
import numpy as np
from keras.utils import image_utils
import cv2
import dlib
from PIL import Image

# PATH TO ALL IMAGES
global basedir, image_paths, target_size
basedir = './Dataset/cartoon_set/'
images_dir = os.path.join(basedir, 'img')
labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./B2/shape_predictor_68_face_landmarks.dat')


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def crop_face(features, filename):# Extracting eye images from pictures
    # cv2 reads images
    img = cv2.imread(images_dir + "\\" + filename + str('.png'))
    img = cv2.resize(img[:, :, ::-1], dsize=(500, 500))

    # Number of faces rects
    rects = detector(img, 0)
    for k, point in enumerate(rects):
        shape = predictor(img, point)
        for num in range(shape.num_parts):
            cv2.circle(img, (shape.parts()[num].x, shape.parts()[num].y), 1, (0, 255, 0), -1)

    # Framing the eye area
    x_eye_left_up = features[37][0]
    y_eye_left_up = features[37][1]
    x_eye_right_down = features[40][0]
    y_eye_right_down = features[40][1]

    #Crop the eye area
    eye_box = (x_eye_left_up, y_eye_left_up, x_eye_right_down, y_eye_right_down)
    eye_crop = Image.fromarray(img).crop(eye_box)
    eye_crop = eye_crop.resize((50, 50))

    # Image to array conversion
    eye_crop = np.asarray(eye_crop)
    return eye_crop


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image


def extract_features_labels():
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extract the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    gender_labels = {line.split('\t')[0]: int(line.split('\t')[1]) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for img_path in image_paths:
            file_name = img_path.split('.')[1].split('.')[0].split('\\')[-1]
            # load image
            img = image_utils.img_to_array(
                image_utils.load_img(img_path,
                                     target_size=target_size,
                                     interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(crop_face(features, file_name))
                all_labels.append(gender_labels[file_name])

    landmark_features = np.array(all_features)
    # print(landmark_features)
    gender_labels = np.array(all_labels)  # simply converts the -1 into 0, so male=0 and female=1
    # print(gender_labels.shape)
    return landmark_features, gender_labels
