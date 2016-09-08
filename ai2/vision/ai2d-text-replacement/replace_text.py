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

from parse_annotation import is_this_text_in_relationship


is_visualize = False
target_rels = ['intraObjectLinkage', 'intraObjectRegionLabel', 'intraObjectLabel', 'intraObjectTextLinkage']


class Rect_attribute:
    is_easy = False
    replacing_color = (0,0,0)
    text_color = (0,0,0)
    bb = []
    tight_bb = []
    text_org = None
    text_to_replace = None

    def __init__(self):
        pass


def crop_with_safe_pad(img, rect, pad=0):
    start_y = max(rect[0][1]-pad, 0)
    start_x = max(rect[0][0]-pad, 0)
    return img[start_y:rect[1][1]+pad, start_x:rect[1][0]+pad, :], [start_x, start_y]  # python is insensitve to outside indexing


def put_homogeneous_patch(img, rect, majority_color, pad=0, do_perturb=False):
    """
    this function modifies the img argument
    """
    if type(pad) == list:
        pad_x = pad[0]
        pad_y = pad[1]
    else:
        pad_x = pad
        pad_y = pad
    # 1. replace patch with homogeneous color
    start_y = max(rect[0][1]-pad_y, 0)
    start_x = max(rect[0][0]-pad_x, 0)
    if not do_perturb:
        img[start_y:rect[1][1]+pad_y, start_x:rect[1][0]+pad_x, :] = majority_color  # python is insensitve to outside indexing
        return
    # 2. replace patch with perturbed color (use start_x and start_y)
    end_y = min(rect[1][1]+pad_y, img.shape[0])
    end_x = min(rect[1][0]+pad_x, img.shape[1])
    #
    replacing_patch = np.ones((end_y-start_y, end_x-start_x, img.shape[2]), dtype='uint8')
    for yy in range(0,replacing_patch.shape[0]):
        for xx in range(0,replacing_patch.shape[1]):
            replacing_patch[yy,xx,:] = np.minimum(np.maximum(majority_color + 5*(np.random.rand(1,3)-0.5), 0), 255).astype('uint8')
    img[start_y:rect[1][1]+pad_y, start_x:rect[1][0]+pad_x, :] = replacing_patch  # python is insensitve to outside indexing


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


def get_tight_bb_with_ocr(patch):
    patch_fn = './temp_patch_%d.png' % int(np.random.rand()*10000)
    cv2.imwrite(patch_fn, patch)
    #-- read in binary for OCR to get tight bb
    img_bin_data = None
    with open(patch_fn, "rb") as f:
        img_bin_data = f.read()
    os.remove(patch_fn)
    request_params = {}
    request_params.update(dict(image=base64.b64encode(img_bin_data).decode('ascii')))
    try:
        res = requests.post('http://vision-ocr.dev.allenai.org/v1/ocr', json=request_params)
    except Exception as e:
        print(e)
        return []
    #
    if res.status_code != 200:
        print("received error process ocr detections [%s]: %s" % (res.status_code, res.content))
        return [[0,0],[0,0]]
    # res.raise_for_status()
    api_detections = res.json()
    # todo: change to elegant python sorting by a class method
    rects = []
    scores = []
    for detection in api_detections['detections']:
        rect_ = [[detection['rectangle'][0]['x'], detection['rectangle'][0]['y']],
                [detection['rectangle'][1]['x'], detection['rectangle'][1]['y']]]
        rects.append(rect_)
        scores.append(detection['score'])
    # sort by the score
    sorted_idx = np.array(scores).argsort()[::-1] # desceding order
    sorted_rects = [rects[i] for i in sorted_idx]
    return sorted_rects


def get_rects_to_replace(fn, img, annotation, cropping_func_ptr):
    rects = []
    text_annotations = annotation['text']  # text regions
    for i, ta in enumerate(text_annotations):
        if not is_this_text_in_relationship(annotation['relationships'], ta, target_rels):
            continue
        rect_attr = Rect_attribute()
        rect = text_annotations[ta]['rectangle']
        # crop the annotated rectangle first
        # img_cropped = crop_with_safe_pad(img, rect, 10)
        img_cropped, start_pt = cropping_func_ptr(img, rect, 10)
        #
        if is_visualize:
            cv2.imshow("cropped", img_cropped)
            cv2.waitKey(1)
        #-- 1. determine if each patch's background is homogeneous color by histogram magnitude
        # - K-means
        img_array = img_cropped.reshape((img_cropped.shape[0] * img_cropped.shape[1], 3))
        clt = KMeans(n_clusters=5)
        clt.fit(img_array)
        hist = centroid_histogram(clt)
        # find the majority color
        majority_hist_idx = np.argmax(hist)
        majority_color = clt.cluster_centers_[majority_hist_idx]
        print("[%s's %d-th patch] majority in histogram: %f. Majority color:" % (fn, i, hist[majority_hist_idx]), majority_color)
        # determine the patch is easy or not (a heuristic criterion)
        if hist[majority_hist_idx] > 0.5:
            rect_attr.is_easy = True
        else:
            rect_attr.is_easy = False
        # Getting tight BB by calling OCR
        rect_attr.bb = rect
        ocr_api_rects = get_tight_bb_with_ocr(img_cropped)
        if len(ocr_api_rects) > 0:
            rect_attr.tight_bb = (np.array([start_pt, start_pt]) + np.array(ocr_api_rects[0])).tolist()
        else:
            rect_attr.tight_bb = rect  # if OCR does not return anything meaningful, just assign the annotated text box
        rect_attr.text_org = text_annotations[ta]['value']
        rect_attr.text_to_replace = text_annotations[ta]['replacementText']
        rect_attr.replacing_color = majority_color
        if majority_color.mean() < 40:
            rect_attr.text_color = (255, 255, 255)
        else:
            rect_attr.text_color = (0, 0, 0)
        #
        rects.append(rect_attr)
    return rects


def get_mask(rects, img, annotation, exclude_arrow=False, exclude_blob=False):
    mask = np.zeros(img.shape[:-1], dtype=img.dtype)
    for rect_attr in rects:
        if not rect_attr.is_easy:
            rect = rect_attr.bb
            mask[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] = 255  # todo: change it to a function call
    # exclude all arrows
    if exclude_arrow:
        for arrow_key in annotation['arrows']:
            arrow_polygon = annotation['arrows'][arrow_key]['polygon']
            mask = cv2.fillConvexPoly(mask, np.array(arrow_polygon, dtype=np.int32), (0))
    # exclude all blobs
    if exclude_blob:
        for arrow_key in annotation['blobs']:
            arrow_polygon = annotation['blobs'][arrow_key]['polygon']
            mask = cv2.fillConvexPoly(mask, np.array(arrow_polygon, dtype=np.int32), (0))
    if is_visualize:
        cv2.imshow("mask wo arrow", mask)
        cv2.waitKey(1)
    return mask


def remove_rectangles(rects, img, use_tight_bb=False):
    img_cp = img.copy()
    for rect_attr in rects:
        if use_tight_bb:
            rect = rect_attr.tight_bb
        else:
            rect = rect_attr.bb
        majority_color = rect_attr.replacing_color
        pad_rect = [5,2]
        put_homogeneous_patch(img_cp, rect, (255,255,255), pad=pad_rect, do_perturb=False) # todo: revert this quick hack (make BG patch with white bg)
        # put rectangle as border
        pad_rect_np = np.array(pad_rect)
        cv2.rectangle(img_cp, tuple(np.maximum(np.array(rect[0])-pad_rect_np, 0)), tuple(np.minimum(np.array(rect[1])+pad_rect_np, img_cp.shape[::-1][1:3])), (0,0,0), 1)
    return img_cp


def restore_arrow_blob(img_org, img_modified, annotation, restore_arrow=True, restore_blob=True):
    """
    restore all blob and arrows (if the text is inside the blob, it is bad)
    """
    mask_temp = np.zeros(img_org.shape, dtype=np.uint8)
    if restore_blob:
        for blob_key in annotation['blobs']:
            blob_polygon = annotation['blobs'][blob_key]['polygon']
            cv2.fillPoly(mask_temp, [np.array(blob_polygon)], (255, 255, 255))
    if restore_arrow:
        for arrow_key in annotation['arrows']:
            arrow_polygon = annotation['arrows'][arrow_key]['polygon']
            cv2.fillPoly(mask_temp, [np.array(arrow_polygon)], (255, 255, 255))
    blob_and_arrow = cv2.bitwise_and(img_org, mask_temp)
    removed_crop   = cv2.bitwise_and(img_modified, 255-mask_temp)
    img = cv2.add(removed_crop, blob_and_arrow)
    #
    if is_visualize:
        cv2.imshow('removed', img)
        cv2.waitKey(1)
    return img


def put_text_in_rects(img_result, rects, img, fn):
    fn_temp = "./temp_%d.png" % int(np.random.rand()*10000)
    cv2.imwrite(fn_temp, img_result * 255)
    surface = cairo.ImageSurface.create_from_png(fn_temp)
    ctx = cairo.Context(surface)
    os.remove(fn_temp)
    # compute text box size by averaging all text box sizes
    heights = []
    for rect_attr in rects:
        rect = rect_attr.bb
        heights.append(rect[1][1] - rect[0][1])
    mean_height = np.median(heights)
    #
    for rect_attr in rects:
        rect = rect_attr.bb
        rect_heights = rect[1][1] - rect[0][1]
        text_color = rect_attr.text_color
        # #- put text by opencv
        # img_result_text_replaced = cv2.putText(img_result_text_replaced, replacement_texts[i], (int((0.6*rect[0][0]+0.4*rect[1][0])), int((0.2*rect[0][1]+0.8*rect[1][1]))),
        #             cv2.FONT_HERSHEY_DUPLEX, 0.4/13.0*float(rect[1][1]-rect[0][1]), text_color)

        #- put text by CAIRO
        ctx.select_font_face('Sans')
        font_height = 0.9*rect_heights
        ctx.set_font_size(font_height)  # em-square height is 90 pixels
        # ctx.move_to(int(0.5*rect[0][0]+0.5*rect[1][0]), int(0.1*rect[0][1]+0.9*rect[1][1]))  # move to point (x, y)

        (x, y, width, height, dx, dy) = ctx.text_extents(rect_attr.text_to_replace)
        ctx.move_to((rect[0][0]+rect[1][0])/2 - width/2, (rect[0][1]+rect[1][1])/2 + height/2)

        ctx.set_source_rgb(text_color[0], text_color[1], text_color[2])  # yellow
        ctx.show_text(rect_attr.text_to_replace)
    #
    ctx.stroke()  # commit to surface
    surface.write_to_png('./replaced/'+ fn)  # write to file


def replace_text_single_image(fn, dataset_path):
    do_inpainting = False

    print("[%s] begins" % fn)
    annotation_fn = os.path.join(dataset_path, 'simple_annotations', fn+'.json')
    with open(annotation_fn) as f:
        annotation = json.loads(f.read())
    #--- read images in multiple formats (due to Cairo)
    # - read img in numpy
    img = cv2.imread(os.path.join(dataset_path, 'images', fn))
    # - read for cairo
    surface = cairo.ImageSurface.create_from_png(os.path.join(dataset_path, 'images', fn))
    ctx = cairo.Context(surface)
    #---
    if is_visualize:
        cv2.imshow("img", img)
        cv2.waitKey(1)

    # 1. get the rectangles to replace the text inside
    rects = get_rects_to_replace(fn, img, annotation, crop_with_safe_pad)
    if len(rects) == 0:
        print('[%s] no rectangles to replace' % fn)
        return

    # 2. generating text mask only for complicated regions
    if do_inpainting:
        mask = get_mask(rects, img, annotation, exclude_arrow=True, exclude_blob=False)

    # 3. put homogeneous patches (remove original text)
    img_modified = remove_rectangles(rects, img, use_tight_bb=False)

    # 4. restore arrow and blob region
    img_result = restore_arrow_blob(img, img_modified, annotation, restore_arrow=False, restore_blob=False)

    # # 4. inpaint with bi-harmonic algorithm
    if do_inpainting:
        print("[%s] inpainting..." % fn)
        img_result = inpaint.inpaint_biharmonic(img_result, mask, multichannel=True)
    else:
        img_result = (img_result * 255).astype('uint8')

    if is_visualize:
        cv2.imshow("removed", img_result)
        cv2.waitKey(1)

    # 5. put text on the cleaned up image
    print("[%s] putting text..." % fn)
    put_text_in_rects(img_result, rects, img, fn)

    # finish
    print("[%s] done" % fn)


if __name__ == '__main__':
    dataset_path = "./ai2d"
    # read list of images in GND category annotation
    with open(os.path.join(dataset_path, "categories.json")) as f:
        file_list = json.loads(f.read())
    # #
    # parallel.multimap(replace_text_single_image, file_list, dataset_path)

    # for fn in file_list:
    #     replace_text_single_image(fn, dataset_path)

    fn = '4647.png' # '636.png' # '4837.png'
    replace_text_single_image(fn, dataset_path)
