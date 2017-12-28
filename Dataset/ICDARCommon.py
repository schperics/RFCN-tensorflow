import cv2

try:
    from . import BoxAwareRandZoom
except:
    import BoxAwareRandZoom

MOD_FACTOR = 64
MIN_BOX = 32
MAX_NUM = 4
SCALE_FACTOR = 4  # int(1024/MOD_FACTOR/MAX_NUM)

# height, width
normalizedRatio = [[4, 4],
                   [3, 4],
                   [4, 3],
                   [3, 3],
                   [3, 2],
                   [2, 4],
                   [4, 2]]


def normalizeImageSize(h, w):
    r = h / w
    minDiff = 999999
    selected = None
    for s in normalizedRatio:
        sr = s[0] / s[1]
        diff = abs(sr - r)
        if diff < minDiff:
            minDiff = diff
            selected = s
    return selected[0] * MOD_FACTOR * SCALE_FACTOR, \
           selected[1] * MOD_FACTOR * SCALE_FACTOR


def preprocessImages(img, iBoxes,
                     normalizeSize=True,
                     randomZoom=True):
    if randomZoom:
        img, iBoxes = BoxAwareRandZoom.randZoom(img,
                                                iBoxes,
                                                keepOriginalRatio=False,
                                                keepOriginalSize=False,
                                                keepBoxes=True)

    sizeMul = (1, 1)
    if normalizeSize:
        h, w, _ = img.shape
        nh, nw = normalizeImageSize(h, w)
        if nh is None or nw is None:
            print("Warning: Invalid image ratio skipping: {},{}".format(h, w))
            return None, None, None

        img = cv2.resize(img, (nw, nh))
        sizeMul = (nh / h, nw / w)

    boxes = []
    categories = []
    for i in range(len(iBoxes)):
        x1, y1, w, h = iBoxes[i]["x"], iBoxes[i]["y"], iBoxes[i]["w"], iBoxes[i]["h"]
        newBox = [int(x1 * sizeMul[1]),
                  int(y1 * sizeMul[0]),
                  int((x1 + w) * sizeMul[1]),
                  int((y1 + h) * sizeMul[0])]
        newBox[0] = max(min(newBox[0], img.shape[1]), 0)
        newBox[1] = max(min(newBox[1], img.shape[0]), 0)
        newBox[2] = max(min(newBox[2], img.shape[1]), 0)
        newBox[3] = max(min(newBox[3], img.shape[0]), 0)

        # TODO
        if (newBox[2] - newBox[0]) >= MIN_BOX and (newBox[3] - newBox[1]) >= MIN_BOX:
            boxes.append(newBox)
            categories.append(0)  # Always text

    return img, boxes, categories
