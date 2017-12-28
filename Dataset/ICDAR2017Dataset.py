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

import codecs
import glob
import os
import random

import cv2
import numpy as np

try:
    from . import BoxAwareRandZoom
except:
    import BoxAwareRandZoom

try:
    from . import ICDARCommon
except:
    import ICDARCommon


class ICDAR2017Dataset:
    def __init__(self, path, set="train", normalizeSize=True, randomZoom=True):
        print(path)
        self.path = path
        self.normalizeSize = normalizeSize
        self.set = set
        self.randomZoom = randomZoom
        self.categories = ['text']

    def init(self):
        image_files = glob.glob(os.path.join(self.path, self.set, "image", "img_*.jpg"))
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
            for l in codecs.open(gtFile, "r", "utf-8"):
                s = l.split(',')
                v = [int(t) for t in s[:8]]
                x = np.array([v[0], v[2], v[4], v[6]])
                y = np.array([v[1], v[3], v[5], v[7]])
                x0 = np.min(x)
                y0 = np.min(y)
                x1 = np.max(x)
                y1 = np.max(y)
                w = (x1 - x0)
                h = (y1 - y0)
                iBoxes.append({
                    "x": int(x0),
                    "y": int(y0),
                    "w": w,
                    "h": h
                })

            img, boxes, categories = ICDARCommon.preprocessImages(img, iBoxes)

            if len(boxes) == 0:
                print("Warning: No boxes on image. Skipping.")
                continue

            boxes = np.array(boxes, dtype=np.float32)
            boxes = np.reshape(boxes, [-1, 4])
            categories = np.array(categories, dtype=np.uint8)

            return img, boxes, categories

    def count(self):
        return len(self.image_ids)


if __name__ == "__main__":
    dataset = ICDAR2017Dataset("/mnt/icdar2017_mlt", set="train")
    dataset.init()
    for id in dataset.image_ids:
        imgFile = os.path.join("/mnt/icdar2017_mlt/train", "image", "{}.jpg".format(id))
        img = cv2.imread(imgFile)
        h, w, c = img.shape
        nh, nw = ICDARCommon.normalizeImageSize(h, w)
        print(imgFile, h, w, h / w, nh, nw)

    print(dataset.count())
