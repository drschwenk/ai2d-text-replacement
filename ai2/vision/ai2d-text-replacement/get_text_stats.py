import os
import json
import numpy as np
import parallel
import base64
import requests

from parse_annotation import is_this_text_in_relationship
from _collections import OrderedDict

target_rels = ['intraObjectLinkage', 'intraObjectRegionLabel', 'intraObjectLabel', 'intraObjectTextLinkage']


def get_text_stat(fn, dataset_path, text_hist):
    annotation_fn = os.path.join(dataset_path, 'simple_annotations', fn+'.json')
    with open(annotation_fn) as f:
        annotation = json.loads(f.read())
    #
    text_annotations = annotation['text']  # text regions
    for i, ta in enumerate(text_annotations):
        if not is_this_text_in_relationship(annotation['relationships'], ta, target_rels):
            continue
        text_val = text_annotations[ta]['value']
        if text_val in text_hist:
            text_hist[text_val] += 1
        else:
            text_hist[text_val] = 1


def plot_dict(d):
    import pylab as pl
    X = np.arange(len(d))
    pl.bar(X, d.values(), align='center', width=0.5)
    pl.xticks(X, list(d.keys()), rotation='vertical')
    ymax = max(d.values()) + 1
    pl.ylim(0, ymax)
    pl.show()


def count_dict_val_greater_thres(d, thr):
    return len([i for i in list(d.values()) if i > thr])


if __name__ == "__main__":
    dataset_path = "./ai2d"
    # read list of images in GND category annotation
    with open(os.path.join(dataset_path, "categories.json")) as f:
        file_list = json.loads(f.read())
    #
    text_hist = {}
    # parallel.multimap(get_text_stat, file_list, dataset_path, text_hist)

    for fn in file_list:
        get_text_stat(fn, dataset_path, text_hist)

    thr = 20
    print('Number of unique texts: %d' % len(text_hist))
    print('Number of unique text appearing more than %d times: %d' % (thr, count_dict_val_greater_thres(text_hist, thr)) )

    print('== sorted histogram ==')
    sorted_d = OrderedDict(sorted(text_hist.items(), key=lambda t: t[1]))
    sorted_d_thr = OrderedDict()
    for w in sorted(sorted_d, key=sorted_d.get, reverse=True):
        if sorted_d[w] < thr:
            continue
        print(w, text_hist[w])
        sorted_d_thr[w] = text_hist[w]

    print('== plotting ==')
    # plot_dict(text_hist)
    plot_dict(sorted_d_thr)