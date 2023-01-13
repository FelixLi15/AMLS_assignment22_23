import os
import numpy as np
from keras.utils import image_utils
import cv2
import dlib
import PIL
from PIL import Image

# PATH TO ALL IMAGES
global basedir, image_paths, target_size
basedir = './Dataset/cartoon_set_test/'
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./B2/shape_predictor_68_face_landmarks.dat')


# how to find frontal human faces in an image using 68 landmarks.  These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.

# The face detector we use is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#     300 faces In-the-wild challenge: Database and results.
#     Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.


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

def crop_face(features,filename):
        # cv2读取图像
        img = cv2.imread(images_dir + "\\" + filename+str('.png'))
        img = cv2.resize(img[:, :, ::-1], dsize=(500, 500))

        # 取灰度
        # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 人脸数 rects
        rects = detector(img, 0)
        for k, point in enumerate(rects):
            shape = predictor(img, point)
            landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
            for num in range(shape.num_parts):
                cv2.circle(img, (shape.parts()[num].x, shape.parts()[num].y), 1, (0, 255, 0), -1)
        # cv2.imshow('image', img)

        # if len(rects) == 0:

        # 眼睛+眉毛
        x_eye_left_up = features[37][0]       # 3  23
        y_eye_left_up = features[37][1]       # 18  18
        x_eye_right_down = features[40][0]    # 92  50
        y_eye_right_down = features[40][1]    # 43  43
        #
        eye_box = (x_eye_left_up, y_eye_left_up, x_eye_right_down, y_eye_right_down)
        eye_crop = Image.fromarray(img).crop(eye_box)
        eye_crop = eye_crop.resize((50,50))

        #eye_crop.save("./eye3" + "/" + filename+str('.png'))
        eye_crop=np.asarray(eye_crop)
        return eye_crop

        # 鼻子
        # x_nose_left_up = 98
        # y_nose_left_up = 130
        # x_nose_right_down = 155
        # y_nose_right_down = 173
        #
        # nose_box = (x_nose_left_up, y_nose_left_up, x_nose_right_down, y_nose_right_down)
        # nose_crop = Image.fromarray(img).crop(nose_box)
        # nose_crop.save("./nose" + "/" + filename)

        # 嘴巴
        # x_mouse_left_up = 90
        # y_mouse_left_up = 173
        # x_mouse_right_down = 160
        # y_mouse_right_down = 210
        #
        # mouse_box = (x_mouse_left_up, y_mouse_left_up, x_mouse_right_down, y_mouse_right_down)
        # mouse_crop = Image.fromarray(img).crop(mouse_box)
        # mouse_crop.save("./mouse" + "/" + filename)

        # else:
        #     for i in range(len(rects)): # 超过一个人脸时
        #         landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
        #         # print(type(landmarks))  # numpy.matrix
        #
        #         # 眼睛+眉毛
        #         x_left_up = landmarks[17, 0]
        #         y_left_up = landmarks[18, 1] - (landmarks[29, 1] - landmarks[28, 1])*2
        #         x_right_down = landmarks[26, 0]
        #         y_right_down = landmarks[28, 1] + (landmarks[29, 1] - landmarks[28, 1])
        #
        #         # 嘴巴+鼻子
        #         # x_left_up = landmarks[4, 0] + (landmarks[48, 0] - landmarks[4, 0]) / 2
        #         # y_left_up = landmarks[29, 1]
        #         # x_right_down = landmarks[54, 0] + (landmarks[12, 0] - landmarks[54, 0]) / 2
        #         # y_right_down = landmarks[57, 1] + (landmarks[8, 1] - landmarks[57, 1]) / 2
        #
        #         # 眼睛+眉毛+鼻子
        #         # x_left_up = landmarks[17, 0]
        #         # y_left_up = landmarks[18, 1] - (landmarks[29, 1] - landmarks[28, 1])*2
        #         # x_right_down = landmarks[26, 0]
        #         # y_right_down = landmarks[33, 1] + (landmarks[51, 1] - landmarks[33, 1]) / 2
        #
        #         # 眼睛+眉毛+鼻子+嘴巴
        #         # x_left_up = landmarks[17, 0]
        #         # y_left_up = landmarks[18, 1] - (landmarks[29, 1] - landmarks[28, 1]) * 2
        #         # x_right_down = landmarks[54, 0] + (landmarks[12, 0] - landmarks[54, 0])
        #         # y_right_down = landmarks[57, 1] + (landmarks[8, 1] - landmarks[57, 1]) / 2
        #
        #         box = (x_left_up, y_left_up, x_right_down, y_right_down)
        #         image_crop = Image.fromarray(img).crop(box)
        #         image_crop.save("./eye3" + "/" + filename)

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
    gender_labels = {line.split('\t')[0] : int(line.split('\t')[1]) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for img_path in image_paths:
            file_name= img_path.split('.')[1].split('.')[0].split('\\')[-1]
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
    #print(landmark_features)
    gender_labels = np.array(all_labels)# simply converts the -1 into 0, so male=0 and female=1
    #print(gender_labels.shape)
    return landmark_features, gender_labels



