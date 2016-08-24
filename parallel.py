import os
import multiprocessing


def multimap(method, iterable, *args):
    # Must use spawn instead of fork or native packages such as cv2 and igraph will cause
    # children to die sporadically causing the Pool to hang
    procs = os.cpu_count()
    #
    multiprocessing.set_start_method('spawn', force=True)
    pool = multiprocessing.Pool(procs)
    pool_data = []
    for i in iterable:
        pool_data.append(tuple([i] + list(args)))
    results = pool.starmap(method, pool_data)
    pool.close()
    pool.join()
    return results