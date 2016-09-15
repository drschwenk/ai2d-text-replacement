import cv2
import os
import json
import numpy as np
import parallel

from parse_annotation import is_this_text_in_relationship


is_visualize = False
target_rels = ['intraObjectLinkage', 'intraObjectRegionLabel', 'intraObjectLabel', 'intraObjectTextLinkage']


def write_gnd_rects_single_image(fn, dataset_path, output_path):
    print("[%s] begins" % fn)
    annotation_fn = os.path.join(dataset_path, 'annotations', fn+'.json')
    with open(annotation_fn) as f:
        annotation = json.loads(f.read())
    #
    img = cv2.imread(os.path.join(dataset_path, 'images', fn))
    im_shape = img.shape
    #
    gnd_vals = []
    text_annotations = annotation['text']  # text regions
    pad_rect_np = np.array([5, 2]) # padding rectangles
    for i, ta in enumerate(text_annotations):
        if not is_this_text_in_relationship(annotation['relationships'], ta, target_rels):
            continue
        rect = text_annotations[ta]['rectangle']
        # pad rect
        rect[0] = np.maximum(np.array(rect[0]) - pad_rect_np, 0)
        rect[1] = np.minimum(np.array(rect[1]) + pad_rect_np, np.array(im_shape[::-1][1:3])-1)
        #
        start_x = int(rect[0][0])
        start_y = int(rect[0][1])
        end_x = int(rect[1][0])
        end_y = int(rect[1][1])
        width = end_x - start_x
        height = end_y - start_y
        assert(width >= 0)
        assert(height >= 0)
        out_dict = {}
        out_dict['rect'] = [start_x, start_y, width, height]
        out_dict['text'] = text_annotations[ta]['value'].lower()
        gnd_vals.append(out_dict)
    # write back to the output file
    output_fn = os.path.join(output_path, fn+'.json')
    with open(output_fn, 'w') as f:
        data = json.dumps(gnd_vals, indent=4)
        f.write(data)
    print("[%s] finished" % fn)


def read_gnd_rects_single_image(fn, json_path):
    output_fn = os.path.join(json_path, fn+'.json')
    with open(output_fn, 'r') as f:
        jsonlist = json.loads(f.read())
    return jsonlist


if __name__ == '__main__':
    dataset_path = "./ai2d"
    output_path = './ai2d_rect_json'
    # read list of images in GND category annotation
    with open(os.path.join(dataset_path, "categories.json")) as f:
        file_list = json.loads(f.read())

    # #
    # parallel.multimap(write_gnd_rects_single_image, file_list, dataset_path, output_path)

    for fn in file_list:
        write_gnd_rects_single_image(fn, dataset_path, output_path)

    print('test reading...')
    rects = {}
    json_path = output_path
    for fn in file_list:
        rects[fn] = read_gnd_rects_single_image(fn, json_path)

    # fn = '4647.png' # '636.png' # '4837.png'
    # replace_text_single_image(fn, dataset_path)
