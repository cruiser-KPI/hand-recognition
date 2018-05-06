import cv2
import numpy as np

import object_classification

model = object_classification.get_trained_model()
label_map = object_classification.LABEL_MAP


def get_boxes(image):
    '''  Return object boxes proposals for image using selective search algorithm '''

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    #print('Total Number of Region Proposals: {}'.format(len(rects)))
    return rects


# Malisiewicz et al.
def non_max_suppression(boxes, overlapThresh=0.2):
    '''  Return reduced number of boxes using non-max suppression algorithm '''

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    return boxes[pick]


def detect_objects(image, min_condifidence, overlap_threshold):
    '''  Return object boxes (together with its class and confidence) detected in image '''

    confidence_levels = {}
    for value in label_map.values():
        confidence_levels[value] = []

    object_size_x, object_size_y = object_classification.INPUT_DATA_SIZE_X, object_classification.INPUT_DATA_SIZE_Y
    for box in get_boxes(image):
        x, y, w, h = box
        max_scale = 1.2
        min_dim = 30
        if w < min_dim or h < min_dim:
            continue
        if w / h < 1 / max_scale or w / h > max_scale:
            continue

        box_image = image[y:y + h, x:x + w]
        box_image = cv2.cvtColor(box_image, cv2.COLOR_BGR2GRAY)

        # resize each box to the size expected by classifier
        box_image = cv2.resize(box_image, (object_size_x, object_size_y)) / 255.0
        box_image = np.reshape(box_image, (-1, object_size_x, object_size_y, 1))

        # predict class of an object in current box
        result = model.predict(box_image)[0].tolist()
        confidence = max(result)
        result_class = result.index(confidence)
        result_label = label_map[result_class]

        # leave only 'left' and 'right' classes
        if not (result_label == 'left' or result_label == 'right'):
            continue
        confidence_levels[result_label].append([x, y, x + w, y + h, confidence * 100, result_class])

    # leave only boxes with confidence higher than min confidence
    high_score_boxes = sorted([x for x in confidence_levels['left'] + confidence_levels['right'] if x[4] > min_condifidence],
                              key=lambda x: x[4])
    resulting_boxes = non_max_suppression(np.array(high_score_boxes, dtype=np.float), overlap_threshold)

    if len(resulting_boxes) > 0:
        resulting_boxes = resulting_boxes.astype(int)
    return resulting_boxes
