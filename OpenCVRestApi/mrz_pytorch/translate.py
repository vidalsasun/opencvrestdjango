import os
import time
import argparse

from collections import defaultdict 

import pytesseract
from pytesseract import Output

import OpenCVRestApi.mrz_pytorch.models.config

import OpenCVRestApi.mrz_pytorch.document_orientation_preprocessing
import OpenCVRestApi.mrz_pytorch.tesseract_preprocessing

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import OpenCVRestApi.mrz_pytorch.models.crnn as crnn
from OpenCVRestApi.mrz_pytorch.models.crnn_run import CRNNReader

import base64

from PIL import Image

import cv2
from skimage import io
import numpy as np
import OpenCVRestApi.mrz_pytorch.craft_utils
import OpenCVRestApi.mrz_pytorch.imgproc
import OpenCVRestApi.mrz_pytorch.file_utils
import json
import zipfile
import imutils

from OpenCVRestApi.mrz_pytorch.craft import CRAFT
from collections import OrderedDict

import logging
logger = logging.getLogger('django.server')


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

"""result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)
"""

def load_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap =  OpenCVRestApi.mrz_pytorch.imgproc.resize_aspect_ratio(image,OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x =  OpenCVRestApi.mrz_pytorch.imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = OpenCVRestApi.mrz_pytorch.craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = OpenCVRestApi.mrz_pytorch.craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = OpenCVRestApi.mrz_pytorch.craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text =  OpenCVRestApi.mrz_pytorch.imgproc.cvt2HeatmapImg(render_img)

    #if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

#if __name__ == '__main__':
def translate(base64img):
    logger.error('1')
    # load net
    net = CRAFT()     # initialize
    logger.error('2')

    #modelfile = os.path.dirname(__file__) + '/' + OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.trained_model
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    modelfile = BASE_DIR + '/computevision/' + OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.trained_model
    
    logger.error('3, modelfile: ')
    logger.error(modelfile)
    logger.error('---------')


    #print('Loading weights from checkpoint (' + OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.trained_model + ')')
    if OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.cuda:
        net.load_state_dict(copyStateDict(torch.load(modelfile)))
    else:
        net.load_state_dict(copyStateDict(torch.load(modelfile, map_location='cpu')))
    logger.error('4')

    if OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    logger.error('5')

    net.eval()

    logger.error('6')

    # LinkRefiner
    refine_net = None
    if OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        #print('Loading weights of refiner from checkpoint (' + OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.refiner_model + ')')
        if OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.refiner_model, map_location='cpu')))

        refine_net.eval()
        OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.poly = True

    t = time.time()
    
    logger.error('7')

    # load data
    crnn=CRNNReader()

    logger.error('8')

    #print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
    #image = imgproc.loadImage(image_path)
    image = readb64(base64img)   

    logger.error('9')

    angle = OpenCVRestApi.mrz_pytorch.document_orientation_preprocessing.detect_angle(image)
    #print(angle)
    
    logger.error('10')

    if angle > 0:
        image = imutils.rotate_bound(image, angle)

    logger.error('11')

    bboxes, polys, score_text = load_net(net, image, OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.text_threshold, OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.link_threshold, OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.low_text, OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.cuda, OpenCVRestApi.mrz_pytorch.models.config.PyTorchtranslateParams.poly, refine_net)
    results = {}
    v = 0

    logger.error('12')

    #traduction_words = []
    traduction_words={}    
    for _, tmp_box in enumerate(bboxes):
                x = int(tmp_box[0][0])
                y = int(tmp_box[0][1])
                w = int(np.abs(tmp_box[0][0] - tmp_box[1][0]))
                h = int(np.abs(tmp_box[0][1] - tmp_box[2][1]))
                tmp_img =  image[y:y+h, x:x+w]
                tmp_img = Image.fromarray(tmp_img.astype('uint8')).convert('L')
                tmp_img = crnn.transformer(tmp_img)
                tmp_img = tmp_img.view(1, *tmp_img.size())
                tmp_img = Variable(tmp_img)
                results['{}'.format(_)] = crnn.get_predictions(tmp_img)
                v = v + 1
                #pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
                custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789< --psm 6'
                translate = pytesseract.image_to_string(image[y:y+h, x:x+w], lang="OCRB", config=custom_config)
                #traduction_words.append(translate)
                traduction_words[v]=translate
                #cv2.imshow("Rotated (Correct)", image[y:y+h, x:x+w])
                #cv2.waitKey(0)
                #bbox_file = result_folder + "bbox/" + str(v) + '.jpg'
                #cv2.imwrite(bbox_file, image[y:y+h, x:x+w])

    logger.error('13')

    return traduction_words
    # save score text
    """filename, file_ext = os.path.splitext(os.path.basename(image_path))
    mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    cv2.imwrite(mask_file, score_text)
    file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)"""
    #print("elapsed time : {}s".format(time.time() - t))
