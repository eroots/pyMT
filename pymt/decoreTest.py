import numpy as np


def make_vec(func):

    def inner(vec):
        made_change = False
        if np.shape(vec)[0] < np.shape(vec)[1]:
            print('I Made it a column!')
            vec = np.transpose(vec)
            made_change = True
        # return func(vec), made_change
        retval = func(vec)
        if made_change:
            retval = np.transpose(retval)
        return retval
    return inner

    # if made_change:
    #     return np.transpose(inner[0])
    # else:
    #     return inner[0]


@make_vec
def add_vecs(vec1):
    to_add = np.ndarray((106, 1))
    to_add[:] += 100000000
    return vec1 + to_add
