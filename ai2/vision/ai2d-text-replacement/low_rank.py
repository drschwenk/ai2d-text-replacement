import numpy as np


def low_rank(img, thres=0.95):
    """
    Low rank by SVD cut
    :param img: input
    :param thres: thresholding from max singular value
    :return: low rank image
    """
    org_type = img.dtype
    img_arr = img.astype('float32')
    U, S, VT = np.linalg.svd(img_arr, full_matrices=False)
    medval = np.median(S)
    maxval = max(S)
    minval = min(S)
    S[S<thres*maxval] = 0
    S = np.diag(S)
    img_arr_lr = np.dot(U, np.dot(S,VT))
    return img_arr_lr.astype(org_type)

