import cv2
import os
import json
import numpy as np
import parallel

from skimage.restoration import inpaint
from sklearn.cluster import KMeans

import base64
import requests

import cairocffi as cairo

from kmedoids import cluster
from low_rank import low_rank


is_visualize = False
target_rels = ['intraObjectLinkage', 'intraObjectRegionLabel', 'intraObjectLabel', 'intraObjectTextLinkage']


def crop_with_safe_pad(img, rect, pad=0):
    start_y = max(rect[0][1]-pad, 0)
    start_x = max(rect[0][0]-pad, 0)
    return img[start_y:rect[1][1]+pad, start_x:rect[1][0]+pad, :]  # python is insensitve to outside indexing


def put_homogeneous_patch(img, rect, majority_color, pad=0):
    """
    this function modifies the img argument
    :param img:
    :param rect:
    :param majority_color:
    :param pad:
    :return:
    """
    start_y = max(rect[0][1]-pad, 0)
    start_x = max(rect[0][0]-pad, 0)
    img[start_y:rect[1][1]+pad, start_x:rect[1][0]+pad, :] = majority_color  # python is insensitve to outside indexing


def put_homogeneous_patch_with_perturbation(img, rect, majority_color, pad=0):
    """
    this function modifies the img argument
    :param img:
    :param rect:
    :param majority_color:
    :param pad:
    :return:
    """
    start_y = max(rect[0][1]-pad, 0)
    start_x = max(rect[0][0]-pad, 0)
    end_y = min(rect[1][1]+pad, img.shape[0])
    end_x = min(rect[1][0]+pad, img.shape[1])
    #
    replacing_patch = np.ones((end_y-start_y, end_x-start_x, img.shape[2]), dtype='uint8')
    for yy in range(0,replacing_patch.shape[0]):
        for xx in range(0,replacing_patch.shape[1]):
            replacing_patch[yy,xx,:] = np.minimum(np.maximum(majority_color + 5*(np.random.rand(1,3)-0.5), 0), 255).astype('uint8')
    img[start_y:rect[1][1]+pad, start_x:rect[1][0]+pad, :] = replacing_patch  # python is insensitve to outside indexing


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, clt.n_clusters + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def simple_mask(mask, rects, replacement_texts, img, annotation):
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
            if is_visualize:
                cv2.imshow('patch', img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0], :])
                cv2.waitKey(1)
        if is_visualize:
            cv2.imshow("mask", mask)
            cv2.waitKey(1)


def simple_mask_wo_arrow(mask, rects, replacement_texts, img, annotation):
    text_annotations = annotation['text']  # text regions
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
    if is_visualize:
        cv2.imshow("mask wo arrow", mask)
        cv2.waitKey(1)


def put_homogeneous_patch_with_tight_bb(img, rect, majority_color, pad=0):
    patch = crop_with_safe_pad(img, rect, pad)
    patch_fn = './temp_patch_%d.png' % int(np.random.rand()*100)
    cv2.imwrite(patch_fn, patch)
    #-- read in binary for OCR to get tight bb
    img_bin_data = None
    with open(patch_fn, "rb") as f:
        img_bin_data = f.read()
    os.remove(patch_fn)
    request_params = {}
    request_params.update(dict(image=base64.b64encode(img_bin_data).decode('ascii')))
    res = requests.post('http://vision-ocr.dev.allenai.org/v1/ocr', json=request_params)
    #
    if res.status_code != 200:
        print("received error process ocr detections [%s]: %s" % (res.status_code, res.content))

    res.raise_for_status()
    api_detections = res.json()
    #
    for detection in api_detections['detections']:
        rect_ = [[detection['rectangle'][0]['x'], detection['rectangle'][0]['y']],
                [detection['rectangle'][1]['x'], detection['rectangle'][1]['y']]]
        put_homogeneous_patch(patch, rect, majority_color, 0)
    #--
    # modify image back
    start_y = max(rect[0][1]-pad, 0)
    start_x = max(rect[0][0]-pad, 0)
    img[start_y:rect[1][1]+pad, start_x:rect[1][0]+pad, :] = patch


def is_this_text_in_relationship(relationship_annots, text_label, target_relationships):
    for relationship in relationship_annots:
        if text_label in relationship and relationship_annots[relationship]['category'] in target_relationships:
            return True
    return False


def simple_mask_wo_arrow_with_homo_patch(mask, rects, replacement_texts, img, annotation):
    img_org = img.copy()
    text_annotations = annotation['text']  # text regions
    for i, ta in enumerate(text_annotations):
        if not is_this_text_in_relationship(annotation['relationships'], ta, target_rels):
            continue
        print("%s [%s] has the relationship" % (ta, text_annotations[ta]['value']))
        rect = text_annotations[ta]['rectangle']
        replacement_text = text_annotations[ta]['replacementText']

        # 1. determine if each patch's background is homogeneous color by histogram magnitude
        img_cropped = crop_with_safe_pad(img, rect, 10)
        #
        if is_visualize:
            cv2.imshow("cropped", img_cropped)
            cv2.waitKey(1)
        # - K-means
        img_array = img_cropped.reshape((img_cropped.shape[0] * img_cropped.shape[1], 3))
        clt = KMeans(n_clusters=5)
        clt.fit(img_array)
        hist = centroid_histogram(clt)
        # find the majority color
        majority_hist_idx = np.argmax(hist)
        majority_color = clt.cluster_centers_[majority_hist_idx]
        print("[%d-th patch] hist on majority: %f" % (i, hist[majority_hist_idx]), majority_color)
        is_easy = False
        if hist[majority_hist_idx] > 0.5:
            is_easy = True
        #
        replacement_texts.append(replacement_text)
        rects.append(rect)
        #
        if not is_easy:
            mask[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] = 255  # todo: change it to a function call
        # 2. modify img with homogeneous color
        # put_homogeneous_patch(img, rect, majority_color, 3)
        # put_homogeneous_patch_with_tight_bb(img, rect, majority_color, 10)
        put_homogeneous_patch_with_perturbation(img, rect, majority_color, 3)
    # restore all blob and arrows (if the text is inside the blob, it is bad)
    mask_temp = np.zeros(img.shape, dtype=np.uint8)
    for blob_key in annotation['blobs']:
        blob = annotation['blobs'][blob_key]['polygon']
        cv2.fillPoly(mask_temp, [np.array(blob)], (255, 255, 255))
    for arrow_key in annotation['arrows']:
        arrow_polygon = annotation['arrows'][arrow_key]['polygon']
        cv2.fillPoly(mask_temp, [np.array(arrow_polygon)], (255, 255, 255))
    blob_and_arrow = cv2.bitwise_and(img_org, mask_temp)
    removed_crop = cv2.bitwise_and(img, 255-mask_temp)
    img = cv2.add(removed_crop, blob_and_arrow)
    #
    if is_visualize:
        cv2.imshow('removed', img)
        cv2.waitKey(1)

    # todo: separate the code here

    # exclude all arrow and blob
    for arrow_key in annotation['arrows']:
        arrow_polygon = annotation['arrows'][arrow_key]['polygon']
        mask = cv2.fillConvexPoly(mask, np.array(arrow_polygon, dtype=np.int32), (0))
    if is_visualize:
        cv2.imshow("mask wo arrow", mask)
        cv2.waitKey(1)


def put_text_in_rects(img_result, rects, replacement_texts, img, fn):
    img_result_text_replaced = img_result.copy()
    fn_temp = "./temp_%d.png" % int(np.random.rand()*10000)
    cv2.imwrite(fn_temp, img_result * 255)
    surface = cairo.ImageSurface.create_from_png(fn_temp)
    ctx = cairo.Context(surface)
    os.remove(fn_temp)
    # figuring out text box size
    heights = []
    for i, rect in enumerate(rects):
        heights.append(rect[1][1] - rect[0][1])
    mean_height = np.median(heights)
    for i, rect in enumerate(rects):
        img_cropped = crop_with_safe_pad(img, rect, 0)
        img_array = img_cropped.reshape((img_cropped.shape[0] * img_cropped.shape[1], 3))
        if img_array.shape[0] == 0:
            continue
        clt = KMeans(n_clusters = 3)
        clt.fit(img_array)
        hist = centroid_histogram(clt)
        majority_hist_idx = np.argmax(hist)
        majority_color = clt.cluster_centers_[majority_hist_idx]  # todo: this color might have been distorted
        text_color = ()
        if majority_color.mean() < 40:
            text_color = (1.0, 1.0, 1.0)
        else:
            text_color = (0.0, 0.0, 0.0)
        # # put text by opencv
        # img_result_text_replaced = cv2.putText(img_result_text_replaced, replacement_texts[i], (int((0.6*rect[0][0]+0.4*rect[1][0])), int((0.2*rect[0][1]+0.8*rect[1][1]))),
        #             cv2.FONT_HERSHEY_DUPLEX, 0.4/13.0*float(rect[1][1]-rect[0][1]), text_color)
        # put text by CAIRO
        ctx.select_font_face('Sans')
        ctx.set_font_size(0.9*mean_height)  # em-square height is 90 pixels
        ctx.move_to( int((0.6*rect[0][0]+0.4*rect[1][0])), int((0.2*rect[0][1]+0.8*rect[1][1])) )  # move to point (x, y) = (10, 90)
        ctx.set_source_rgb(text_color[0], text_color[1], text_color[2])  # yellow
        ctx.show_text(replacement_texts[i])
    #
    ctx.stroke()  # commit to surface
    surface.write_to_png('./replaced/'+ fn)  # write to file


def replace_text_single_image(fn, dataset_path):
    annotation_fn = os.path.join(dataset_path, 'simple_annotations', fn+'.json')
    with open(annotation_fn) as f:
        annotation = json.loads(f.read())
    # read in numpy
    img = cv2.imread(os.path.join(dataset_path, 'images', fn))
    # read in binary for OCR
    img_bin_data = None
    with open(os.path.join(dataset_path, 'images', fn), "rb") as f:
        img_bin_data = f.read()
    # read for cairo
    surface = cairo.ImageSurface.create_from_png(os.path.join(dataset_path, 'images', fn))
    ctx = cairo.Context(surface)
    # mask for inpainting
    mask = np.zeros(img.shape[:-1], dtype=img.dtype)
    replacement_texts = []
    rects = []

    if is_visualize:
        cv2.imshow("img", img)
        cv2.waitKey(1)

    #---- generate text mask
    # # Appr 1. generating text mask - approach 1
    # simple_mask(mask, rects, replacement_texts, img, annotation)
    #
    # # Appr 2. generating text mask - approach 2: use tight BB using OCR API
    # mask_with_tight_bb(mask, rects, img_bin_data, replacement_texts, img, annotation)

    # # Appr 3. generating text mask - approach 3: use lose BB except arrow region
    # simple_mask_wo_arrow(mask, rects, replacement_texts, img, annotation)

    # Appr 4. generating text mask only for complicated regions
    simple_mask_wo_arrow_with_homo_patch(mask, rects, replacement_texts, img, annotation)
    # ----

    # inpaint with bi-harmonic algorithm
    print("inpainting %s start..." % fn)
    img_result = inpaint.inpaint_biharmonic(img, mask, multichannel=True)
    print("inpainting %s finished" % fn)
    #
    if is_visualize:
        cv2.imshow("removed", img_result)
        cv2.waitKey(1)
    #
    put_text_in_rects(img_result, rects, replacement_texts, img, fn)



if __name__ == '__main__':
    dataset_path = "./ai2d"
    # read list of images in GND category annotation
    with open(os.path.join(dataset_path, "categories.json")) as f:
        file_list = json.loads(f.read())
    #
    parallel.multimap(replace_text_single_image, file_list, dataset_path)

    # for fn in file_list:
    #     replace_text_single_image(fn, dataset_path)

    # fn = '2089.png'
    # replace_text_single_image(fn, dataset_path)