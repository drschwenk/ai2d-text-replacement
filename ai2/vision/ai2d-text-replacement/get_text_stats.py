import os
import json
import numpy as np
import parallel
import re
import pickle
import string

from parse_annotation import is_this_text_in_relationship
from _collections import OrderedDict


target_rels = ['intraObjectLinkage', 'intraObjectRegionLabel', 'intraObjectLabel', 'intraObjectTextLinkage']


def get_text_stat(fn, dataset_path, text_hist, pattern):
    annotation_fn = os.path.join(dataset_path, 'annotations', fn+'.json')
    with open(annotation_fn) as f:
        annotation = json.loads(f.read())
    #
    text_annotations = annotation['text']  # text regions
    for i, ta in enumerate(text_annotations):
        if not is_this_text_in_relationship(annotation['relationships'], ta, target_rels):
            continue
        text_val = text_annotations[ta]['value']
        #-- basic text normalization
        text_val = text_val.strip()  # remove trailing space
        text_val = text_val.lower()  # lower casing
        text_val = pattern.sub('', text_val)  # remove stop word (todo: check if it's necessary)
        if len(text_val) < 2:  # if # of characters in the text is less than 2 (supress unwords)
            continue
        # if it has a special character, do not put it into the dictionary
        if re.search('^[a-zA-Z]*$', text_val) == None:
            continue
        #--
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

    # get stop words
    cachedStopWords = pickle.load(open('cachedStopWords.p', 'rb'))
    pattern = re.compile(r'\b(' + r'|'.join(cachedStopWords) + r')\b\s*')

    for fn in file_list:
        get_text_stat(fn, dataset_path, text_hist, pattern)

    thr = 20
    print('Number of unique texts: %d' % len(text_hist))
    print('Number of unique text appearing more than %d times: %d' % (thr, count_dict_val_greater_thres(text_hist, thr)))

    print('== sorted histogram ==')
    sorted_d = OrderedDict(sorted(text_hist.items(), key=lambda t: t[1]))
    sorted_d_thr = OrderedDict()
    for i, w in enumerate(sorted(sorted_d, key=sorted_d.get, reverse=True)):
        # if sorted_d[w] < thr:
        #     continue
        print(w, ':', text_hist[w])
        sorted_d_thr[w] = text_hist[w]

    print('== plotting ==')
    # plot_dict(text_hist)
    plot_dict(sorted_d_thr)

    print('== save lists ==')
    print('save the list for all')
    with open('classnames_all.txt', 'w') as f:
        for i, key in enumerate(sorted_d_thr):
            f.write(str(i) + ', ' + key +'\n')

    print('save the list for all with frequency')
    with open('classnames_all_w_freq.txt', 'w') as f:
        for i, key in enumerate(sorted_d_thr):
            f.write(str(i) + ', ' + key + ', ' + str(sorted_d_thr[key]) + '\n')

    # 200
    print('save the list for top 199 in regard to frequency')
    with open('classnames_200.txt', 'w') as f:
        cnt = 0
        for i, key in enumerate(sorted_d_thr):
            if cnt == 199:
                break
            cnt += 1
            f.write(str(i) + ', ' + key + '\n')
        f.write('199, unknown\n')

    print('save the list for top 199 with frequency')
    with open('classnames_200_w_freq.txt', 'w') as f:
        cnt = 0
        for i, key in enumerate(sorted_d_thr):
            if cnt == 199:
                break
            cnt += 1
            f.write(str(i) + ', ' + key + ', ' + str(sorted_d_thr[key]) + '\n')
        f.write('199, unknown, -1\n')


    # 50
    print('save the list for top 49 in regard to frequency')
    with open('classnames_50.txt', 'w') as f:
        cnt = 0
        for i, key in enumerate(sorted_d_thr):
            if cnt == 49:
                break
            cnt += 1
            f.write(str(i) + ', ' + key + '\n')
        f.write('49, unknown\n')

    print('save the list for top 49 with frequency')
    with open('classnames_50_w_freq.txt', 'w') as f:
        cnt = 0
        for i, key in enumerate(sorted_d_thr):
            if cnt == 49:
                break
            cnt += 1
            f.write(str(i) + ', ' + key + ', ' + str(sorted_d_thr[key]) + '\n')
        f.write('49, unknown, -1\n')


    # copy it to the ai2d_meta directory
    print('copy the class name to metadata directory of ai2d in NN training folder')
    os.system('cp ./classnames_*.txt /Users/jonghyunc/playground/answer-replaced-text/ai2d_meta/')
