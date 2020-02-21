import cv2
import os, logging, sys, time
from wide_resnet import WideResNet
import numpy as np
from pyagender import PyAgender


class Agender(object):
    '''
    PixelateCore would build a classifier object
    which can called on multiple images
    '''

    def __init__(self, cascadePath=None, odir=None):
        '''
        Constructor for PixelCore
        '''
        self.cascPath = os.path.join(os.getcwd(),
                                     'models',
                                     'haarcascades',
                                     'haarcascade_frontalface_extended.xml')

        resnetwts = os.path.join(cascadePath, 'models', 'weights.18-4.06.hdf5')
        if cascadePath:
            self.cascPath = os.path.join(os.path.abspath(cascadePath),
                                         'models',
                                         'haarcascades',
                                         'haarcascade_frontalface_alt.xml')

        print('Loading HAAR-Cascade file: %s' % (self.cascPath))
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        self.face_size = 32
        self.agender = PyAgender()

    def readGrayScaleImage(self, imgpath=None):
        """
        Reads Image File, checka and converts to grayscale if
        needed

        :param imgpath: path to the image
        :return:
        """
        if imgpath:
            imgpath = imgpath.strip()
        else:
            return

        frame = cv2.imread(imgpath)
        self.frame = cv2.imread(imgpath)
        # check if grayscale conversion is needed
        grayScale = False
        self.grayFrame = self.frame
        if len(self.frame) < 2:
            # strating image is grayscale
            grayScale = True
            logging.info("Image %s is grayscale" % (imgpath))
        else:
            # strating image is color, grayscale conversion is needed
            logging.info("Converting %s to grayscale" % (imgpath))
            self.grayFrame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

    def crop_face(self, imgarray, section, margin=40, size=64):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w, h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w - 1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h - 1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    faces = self.agender.detect_genders_ages(frame)
    for face in faces:
        left = face['left']
        top = face['right']
        cv2.rectangle(frame, (face['top'], face['left']),
                      (face['top'] + face['width'],
                       face['left'] + face['height']), (0, 255, 0),
                      thickness=5)

    def facedetectFrame(self, frame=None):
        """
        :return:
        """
        faces = self.faceCascade.detectMultiScale(
            self.grayFrame,
            scaleFactor=1.2,
            minNeighbors=10,
            minSize=(self.face_size, self.face_size)
        )

    def processImageFrame(self):
        """
        Process Image Frame
        :return:
        """
        ts2 = self.epoch()
        pfnamex = "{}-agender.JPG".format(ts2)
        wwwbase = os.getcwd()
        odir = os.path.join(wwwbase, 'agender')
        if not os.path.exists(odir):
            logging.info('Creating output directory, %s' % (odir))
            os.makedirs(odir)
        pfname = os.path.join(wwwbase, 'agender', pfnamex)
        logging.info('Writing image file, %s' % (pfname))
        cv2.imwrite(pfname, frame)
        return

    def epoch(self):
        """
        Returns Unix Epoch
        """
        epc = int(time.time() * 1000)

        return epc


if __name__ == '__main__':
    print("hello .. ")
    capth = os.getcwd()
    wwwbase = os.path.join(capth, 'ui_www')
    ag = Agender(cascadePath=capth)
    # ag.facedetectFrame(imgpath='/Users/navendusinha/Downloads/img002.jpg',
    #                   wwwbase=capth)
    ag.facedetectFrame(imgpath='/Users/navendusinha/Downloads/IMG_6173.jpg')
