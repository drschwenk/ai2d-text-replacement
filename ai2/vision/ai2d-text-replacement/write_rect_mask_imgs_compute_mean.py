import cv2
import os
import json
import numpy as np
import parallel

from parse_annotation import is_this_text_in_relationship


is_visualize = False
target_rels = ['intraObjectLinkage', 'intraObjectRegionLabel', 'intraObjectLabel', 'intraObjectTextLinkage']


def compute_rect_mask_single_image(fn, dataset_path, output_path, img_cnt, avg_mask, write_to_file=False):
    print("[%s] begins" % fn)
    annotation_fn = os.path.join(dataset_path, 'annotations', fn+'.json')
    with open(annotation_fn) as f:
        annotation = json.loads(f.read())
    #
    img = cv2.imread(os.path.join(dataset_path, 'images', fn))
    im_shape = img.shape
    #
    text_annotations = annotation['text']  # text regions
    pad_rect_np = np.array([5, 2]) # padding rectangles
    cnt = 0
    img_size = avg_mask.shape
    for ta in text_annotations:
        mask_img = np.zeros(im_shape[0:2], dtype=np.uint8)
        if not is_this_text_in_relationship(annotation['relationships'], ta, target_rels):
            continue
        rect = text_annotations[ta]['rectangle']
        # pad rect
        rect[0] = np.maximum(np.array(rect[0]) - pad_rect_np, 0)
        rect[1] = np.minimum(np.array(rect[1]) + pad_rect_np, im_shape[::-1][1:3])
        #
        mask_img[rect[0][0]:rect[1][0], rect[0][1]:rect[1][1]] = 255
        # resize to vgg-16 format
        mask_img_resize = cv2.resize(mask_img.astype(np.float32), img_size, interpolation=cv2.INTER_LINEAR)
        if write_to_file:
            output_fn = os.path.join(output_path, fn + '_rect' + str(cnt) + '.png')
            cv2.imwrite(output_fn, mask_img_resize)
        avg_mask = avg_mask*float(img_cnt)/float(img_cnt+1) + mask_img_resize/float(img_cnt+1)
        cv2.imshow('avg_mask', avg_mask/10)
        cv2.waitKey(1)
        img_cnt += 1
        cnt += 1
    print("[%s] finished" % fn)
    return avg_mask


if __name__ == '__main__':
    dataset_path = "./ai2d"
    output_path = './ai2d_rect_mask_images'
    # read list of images in GND category annotation
    with open(os.path.join(dataset_path, "categories.json")) as f:
        file_list = json.loads(f.read())

    # #
    # parallel.multimap(write_gnd_rects_single_image, file_list, dataset_path, output_path)

    avg_mask = np.zeros((256, 256), dtype=np.float32)
    img_cnt = np.array(0)
    for i, fn in enumerate(file_list):
        print(i, "/", len(file_list), "]", fn)
        avg_mask = compute_rect_mask_single_image(fn, dataset_path, output_path, img_cnt, avg_mask, write_to_file=False)

    print("min:", avg_mask.min(), ", max:", avg_mask.max(), ', mean:', avg_mask.mean())
    avg_mask.save('./avg_mask.np')

    import pdb
    pdb.set_trace()
    print(avg_mask)

    # fn = '4647.png' # '636.png' # '4837.png'
    # replace_text_single_image(fn, dataset_path)
