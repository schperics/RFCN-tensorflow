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

import random
import numpy as np
import cv2
import glob
import os
from . import BoxAwareRandZoom


class ICDAR2013Dataset:
    def __init__(self, path, set="train", normalizeSize=True, randomZoom=True):
        print(path)
        self.path = path
        self.normalizeSize = normalizeSize
        self.set = set
        self.randomZoom = randomZoom
        self.categories = ['text']

    def init(self):
        image_files = glob.glob(os.path.join(self.path, self.set, "image", "*.jpg"))
        self.image_ids = [os.path.basename(f).split('.')[0] for f in image_files]

    def getCaptions(self, categories):
        if categories is None:
            return None

        res = []
        if isinstance(categories, np.ndarray):
            categories = categories.tolist()

        for c in categories:
            res.append(self.categories[c])

        return res

    def load(self):
        while True:
            imgId = random.choice(self.image_ids)
            imgFile = os.path.join(self.path, self.set, "image", "{}.jpg".format(imgId))
            gtFile = os.path.join(self.path, self.set, "gt", "gt_{}.txt".format(imgId))
            img = cv2.imread(imgFile)

            if img is None:
                print("ERROR: Failed to load " + imgFile)
                continue

            sizeMul = 1.0
            padTop = 0
            padLeft = 0

            iBoxes = []
            for l in open(gtFile, "rt"):
                s = l.split(' ')
                x0 = int(s[0])
                y0 = int(s[1])
                x1 = int(s[2])
                y1 = int(s[3])
                w = (x1 - x0)
                h = (y1 - y0)
                iBoxes.append({
                    "x": int(x0),
                    "y": int(y0),
                    "w": w,
                    "h": h
                })

            if self.randomZoom:
                img, iBoxes = BoxAwareRandZoom.randZoom(img,
                                                        iBoxes,
                                                        keepOriginalRatio=False,
                                                        keepOriginalSize=False,
                                                        keepBoxes=True)

            if self.normalizeSize:
                sizeMul = 640.0 / min(img.shape[0], img.shape[1])
                img = cv2.resize(img, (int(img.shape[1] * sizeMul), int(img.shape[0] * sizeMul)))

            m = img.shape[1] % 32
            if m != 0:
                padLeft = int(m / 2)
                img = img[:, padLeft: padLeft + img.shape[1] - m]

            m = img.shape[0] % 32
            if m != 0:
                m = img.shape[0] % 32
                padTop = int(m / 2)
                img = img[padTop: padTop + img.shape[0] - m]

            if img.shape[0] < 256 or img.shape[1] < 256:
                print("Warning: Image to small, skipping: " + str(img.shape))
                continue

            boxes = []
            categories = []
            for i in range(len(iBoxes)):
                x1, y1, w, h = iBoxes[i]["x"], iBoxes[i]["y"], iBoxes[i]["w"], iBoxes[i]["h"]
                newBox = [int(x1 * sizeMul) - padLeft,
                          int(y1 * sizeMul) - padTop,
                          int((x1 + w) * sizeMul) - padLeft,
                          int((y1 + h) * sizeMul) - padTop]
                newBox[0] = max(min(newBox[0], img.shape[1]), 0)
                newBox[1] = max(min(newBox[1], img.shape[0]), 0)
                newBox[2] = max(min(newBox[2], img.shape[1]), 0)
                newBox[3] = max(min(newBox[3], img.shape[0]), 0)

                # TODO
                if (newBox[2] - newBox[0]) >= 16 and (newBox[3] - newBox[1]) >= 16:
                    boxes.append(newBox)
                    categories.append(0)  # Always text

            if len(boxes) == 0:
                print("Warning: No boxes on image. Skipping.")
                continue

            boxes = np.array(boxes, dtype=np.float32)
            boxes = np.reshape(boxes, [-1, 4])
            categories = np.array(categories, dtype=np.uint8)

            return img, boxes, categories

    def count(self):
        return len(self.image_ids)
