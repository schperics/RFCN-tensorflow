#!/usr/bin/python
#
# Copyright 2017 Robert Csordas. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

import argparse
import errno
import glob
import os

import cv2
import numpy as np
import tensorflow as tf

from BoxResnet import BoxResnet
from Utils import CheckpointLoader
from Utils import PreviewIO
from Visualize import Visualize
from icdar2013_eval import script as eval

parser = argparse.ArgumentParser(description="RFCN tester")
parser.add_argument('-gpu', type=str, default="0", help='Train on this GPU(s)')
parser.add_argument('-d', type=str, help='Network checkpoint file')
parser.add_argument('-i', type=str, help='Input file.')
parser.add_argument('-o', type=str, default="", help='Write output here.')
parser.add_argument('-p', type=int, default=1, help='Show preview')
parser.add_argument('-threshold', type=float, default=0.5, help='Detection threshold')
parser.add_argument('-delay', type=int, default=-1,
                    help='Delay between frames in visualization. -1 for automatic, 0 for wait for keypress.')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

categories = ['text']

palette = Visualize.Palette(len(categories))

image = tf.placeholder(tf.float32, [None, None, None, 3])
net = BoxResnet(image, len(categories), name="boxnet")

boxes, scores, classes = net.getBoxes(scoreThreshold=opt.threshold)

input = PreviewIO.PreviewInput(opt.i)
output = PreviewIO.PreviewOutput(opt.o, input.getFps())


def preprocessInput(img):
    def calcPad(size):
        m = size % 64
        p = int(m / 2)
        s = size - m
        return s, p

    zoom = max(640.0 / img.shape[0], 640.0 / img.shape[1])
    img = cv2.resize(img, (int(zoom * img.shape[1]), int(zoom * img.shape[0])))

    if img.shape[0] % 64 != 0:
        s, p = calcPad(img.shape[0])
        img = img[p:p + s]

    if img.shape[1] % 64 != 0:
        s, p = calcPad(img.shape[1])
        img = img[:, p:p + s]

    return img, zoom


with tf.Session() as sess:
    with open(os.path.join(opt.d, "score.txt"), "wt") as score_file:
        for idx_file in glob.glob(os.path.join(opt.d, "save", "model-*.index")):
            model_file = idx_file.split(".")[0]
            if not CheckpointLoader.loadCheckpoint(sess, None, model_file, ignoreVarsInFileNotInSess=True):
                print("Failed to load network. {}", model_file)
                continue
            model = model_file.split("/")[-1]
            input = PreviewIO.PreviewInput(opt.i)

            while True:
                img = input.get()
                if img is None:
                    break

                img, zoom = preprocessInput(img)

                rBoxes, rScores, rClasses = sess.run([boxes, scores, classes],
                                                     feed_dict={image: np.expand_dims(img, 0)})

                num = input.getName().split('.')[0]

                res_path = os.path.join(opt.d, "result", model)
                try:
                    os.makedirs(res_path)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                res_path = os.path.join(opt.d, 'result', model)
                res_filename = os.path.join(res_path, "res_img_{}.txt".format(num))
                with open(res_filename, "wt") as f:
                    print("{}: rboxes = {}".format(res_filename, len(rBoxes)))
                    for rBox in rBoxes:
                        f.write('{},{},{},{}\n'.format(int(rBox[0] / zoom),
                                                       int(rBox[1] / zoom),
                                                       int(rBox[2] / zoom),
                                                       int(rBox[3] / zoom)))

                if opt.p == 1:
                    res = Visualize.drawBoxes(img, rBoxes, rClasses, [categories[i] for i in rClasses.tolist()],
                                              palette,
                                              scores=rScores)

                    output.put(input.getName(), res)

            res = eval.eval_result('icdar2013_eval/gt.zip', res_path)['method']
            res_text = ("{} {} {} {}".format(model, res['hmean'], res['recall'], res['precision']))
            print(res_text)
            score_file.write(res_text + "\n")
            score_file.flush()
