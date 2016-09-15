import cv2
import os
import json
import numpy as np
import parallel

from parse_annotation import is_this_text_in_relationship


is_visualize = False
target_rels = ['intraObjectLinkage', 'intraObjectRegionLabel', 'intraObjectLabel', 'intraObjectTextLinkage']


def find_all_destinations_with_text(ta, annotation):
    """
    'dest' means {'arrow': <arrow_polygon>, 'dest_polygon': <destination's polygon>}
    :param ta: text annotation
    :param annotation: all annotation information
    :return: [{'arrow': <arrow_polygon>, 'dest_blob': <blob_polygon>}]
    """
    dests = []
    for rel in annotation['relationships']:
        this_rel = annotation['relationships'][rel]
        arrow = None
        if ta == this_rel['origin'] or ta == this_rel['destination']:
            if 'connector' in this_rel:
                arrow = annotation['arrows'][this_rel['connector']]['polygon']
            dest_name = this_rel['destination']
            rel_type = this_rel['category']
            if dest_name[0] == 'B':
                dest_polygon = annotation['blobs'][dest_name]['polygon']
            else:
                continue
            out_dict = {'arrow_polygon': arrow, 'dest_polygon': dest_polygon, 'relationship_type': rel_type}
            dests.append(out_dict)
    return dests



def write_gnd_rects_single_image(fn, dataset_path, output_path):
    """
    Write the arrows and target blob polygon only if the text is an origin of the relationship
    :param fn:
    :param dataset_path:
    :param output_path:
    :return:
    """
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
        dests = find_all_destinations_with_text(ta, annotation)
        if len(dests) > 1:
            print(dests)
        out_dict = {}
        out_dict['text'] = text_annotations[ta]['value'].lower()
        out_dict['destinations'] = dests
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
    output_path = './ai2d_arrow_json'
    # read list of images in GND category annotation
    with open(os.path.join(dataset_path, "categories.json")) as f:
        file_list = json.loads(f.read())

    # #
    # parallel.multimap(write_gnd_rects_single_image, file_list, dataset_path, output_path)

    for fn in file_list:
        write_gnd_rects_single_image(fn, dataset_path, output_path)

    # fn = '4212.png'
    # write_gnd_rects_single_image(fn, dataset_path, output_path)

    print('test reading...')
    rects = {}
    json_path = output_path
    for fn in file_list:
        rects[fn] = read_gnd_rects_single_image(fn, json_path)

    # fn = '4647.png' # '636.png' # '4837.png'
    # replace_text_single_image(fn, dataset_path)
