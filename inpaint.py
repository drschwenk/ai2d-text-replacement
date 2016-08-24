import cv2
import os
import json
import numpy as np
import parallel

from skimage.restoration import inpaint

import base64
import requests

is_visualize = True

def low_rank(img):
    org_type = img.dtype
    img_arr = img.astype('float32')
    U, S, VT = np.linalg.svd(img_arr, full_matrices=False)
    medval = np.median(S)
    maxval = max(S)
    minval = min(S)
    S[S<0.95*maxval] = 0
    S = np.diag(S)
    img_arr_lr = np.dot(U, np.dot(S,VT))
    return img_arr_lr.astype(org_type)


def simple_mask(mask, rects, replacement_texts, img, annotation):
    if is_visualize:
        cv2.imshow("img", img)
        cv2.waitKey(1)
        #
        disp_win_name = "cropped"
        cv2.namedWindow(disp_win_name)
        disp_win_name2 = "removed"
        cv2.namedWindow(disp_win_name2)
        cv2.namedWindow("mask")

    text_annotations = annotation['text']
    for ta in text_annotations:
        rect = text_annotations[ta]['rectangle']
        replacement_text = text_annotations[ta]['replacementText']
        #
        replacement_texts.append(replacement_text)
        rects.append(rect)
        #
        cropped_img = img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        if is_visualize:
            cv2.imshow("cropped", cropped_img)
            cv2.waitKey(1)
        #
        mask[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] = 255



def mask_with_tight_bb(mask, rects, img_bin_data, replacement_texts, img, annotation):
    if is_visualize:
        cv2.imshow("img", img)
        cv2.waitKey(1)
        #
        disp_win_name = "cropped"
        cv2.namedWindow(disp_win_name)
        disp_win_name2 = "removed"
        cv2.namedWindow(disp_win_name2)
        cv2.namedWindow("mask")

    text_annotations = annotation['text']
    for ta in text_annotations:
        rect = text_annotations[ta]['rectangle']
        replacement_text = text_annotations[ta]['replacementText']
        #
        replacement_texts.append(replacement_text)
        rects.append(rect)
        #
        cropped_img = img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        if is_visualize:
            cv2.imshow("cropped", cropped_img)
            cv2.waitKey(1)

        request_params = {}
        request_params.update(dict(image=base64.b64encode(img_bin_data).decode('ascii')))
        res = requests.post('http://vision-ocr.dev.allenai.org/v1/ocr', json=request_params)
        #
        if res.status_code != 200:
            print("received error process ocr detections for %s [%s]: %s" % (fn, res.status_code, res.content))

        res.raise_for_status()
        api_detections = res.json()
        #
        for detection in api_detections['detections']:
            rect = [[detection['rectangle'][0]['x'], detection['rectangle'][0]['y']],
                    [detection['rectangle'][1]['x'], detection['rectangle'][1]['y']]]
            mask[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] = 255
            cv2.imshow('patch', img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0], :])
            cv2.waitKey(1)
        if is_visualize:
            cv2.imshow("mask", mask)
            cv2.waitKey(1)


def simple_mask_wo_arrow(mask, rects, replacement_texts, img, annotation):
    # text regions
    text_annotations = annotation['text']
    for ta in text_annotations:
        rect = text_annotations[ta]['rectangle']
        replacement_text = text_annotations[ta]['replacementText']
        #
        replacement_texts.append(replacement_text)
        rects.append(rect)
        #
        cropped_img = img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        if is_visualize:
            cv2.imshow("cropped", cropped_img)
            cv2.waitKey(1)
        mask[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] = 255
    # exclude arrow
    for arrow_key in annotation['arrows']:
        arrow_polygon = annotation['arrows'][arrow_key]['polygon']
        mask = cv2.fillConvexPoly(mask, np.array(arrow_polygon, dtype=np.int32), (0))
    cv2.namedWindow("mask wo arrow")
    cv2.imshow("mask wo arrow", mask)
    cv2.waitKey(1)


def replace_single_image(fn, dataset_path):
    annotation_fn = os.path.join(dataset_path, 'annotations', fn+'.json')
    with open(annotation_fn) as f:
        annotation = json.loads(f.read())
    # read in numpy
    img = cv2.imread(os.path.join(dataset_path, 'images', fn))
    # read in binary
    img_bin_data = None
    with open(os.path.join(dataset_path, 'images', fn), "rb") as f:
        img_bin_data = f.read()
    #
    mask = np.zeros(img.shape[:-1], dtype=img.dtype)
    replacement_texts = []
    rects = []

    if is_visualize:
        cv2.imshow("img", img)
        cv2.waitKey(1)
        #
        disp_win_name = "cropped"
        cv2.namedWindow(disp_win_name)
        disp_win_name2 = "removed"
        cv2.namedWindow(disp_win_name2)
        cv2.namedWindow("mask")

    # # # generating text mask - approach 1
    # simple_mask(mask, rects, replacement_texts, img, annotation)
    #
    # # generating text mask - approach 2: use tight BB using OCR API
    # mask_with_tight_bb(mask, rects, img_bin_data, replacement_texts, img, annotation)

    # generating text mask - approach 3: use lose BB except arrow region
    simple_mask_wo_arrow(mask, rects, replacement_texts, img, annotation)


    # use bi-harmonic inpainting algorithm
    print("inpainting %s start..." % fn)
    img_result = inpaint.inpaint_biharmonic(img, mask, multichannel=True)
    print("inpainting finished")
    #
    cv2.imshow("removed", img_result)
    cv2.waitKey(1)
    #
    img_result_text_replaced = img_result.copy()
    for i, rect in enumerate(rects):
        img_result_text_replaced = cv2.putText(img_result_text_replaced, replacement_texts[i], (int((0.6*rect[0][0]+0.4*rect[1][0])), int((0.2*rect[0][1]+0.8*rect[1][1]))),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4/13.0*float(rect[1][1]-rect[0][1]), (0, 0, 0))
    cv2.imshow("replaced", img_result_text_replaced)
    cv2.waitKey(1)
    # import pdb
    # pdb.set_trace()
    # print(img_result_text_replaced.dtype)
    cv2.imwrite('./replaced/'+fn, img_result_text_replaced*255)



if __name__ == '__main__':
    dataset_path = "./ai2d"
    # read list of images in GND category annotation
    with open(os.path.join(dataset_path, "categories.json")) as f:
        file_list = json.loads(f.read())
    #
    parallel.multimap(replace_single_image, file_list, dataset_path)

    # for fn in file_list:
    #     replace_single_image(fn, dataset_path)