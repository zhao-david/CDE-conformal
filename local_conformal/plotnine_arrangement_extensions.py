# This code was motivated by:
# https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
# and discussions on plotnine github to combine image

import io
from PIL import Image
import plotnine as p9
import warnings
import numpy as np


def gg2img(gg):
    """
    Convert plotnine ggplot figure to PIL Image and return it

    Arguments:
    ----------
    gg: plotnine ggplot object

    Returns:
    --------
    PIL.Image coversion of the ggplot object

    Details:
    --------
    Associated with the stackoverflow question here: https://stackoverflow.com/questions/8598673/how-to-save-a-pylab-figure-into-in-memory-file-which-can-be-read-into-pil-image/8598881
    """

    buf = io.BytesIO()
    gg.save(buf, format= "png")
    buf.seek(0)
    img = Image.open(buf)
    return img

def h_concat(img_list, reshape = True):
    """
    Horizontally concat PIL.Image images

    Arguments:
    ----------
    img_list: list of PIL.Image images
    reshape: boolean, if we should reshape images to be the same size

    Returns:
    --------
        single PIL.Image image that has the original images in a horizontal row

    Example:
    -------
    test_list = [my_img, my_img, my_img]

    h_concat(test_list)
    """
    if reshape:
        min_shape = sorted( [(np.sum(i.size), i.size ) for i in img_list])[0][1]
        imgs_comb = np.hstack( [np.asarray( i.resize(min_shape) ) for i in img_list] )
    else:
        imgs_comb = np.hstack(img_list)
    imgs_comb = Image.fromarray( imgs_comb)
    return imgs_comb

def v_concat(img_list, reshape = True):
    """
    Vertically concat PIL.Image images

    Arguments:
    ----------
    img_list: list of PIL.Image images
    reshape: boolean, if we should reshape images to be the same size

    Returns:
    --------
        single PIL.Image image that has the original images in a vertically col

    Example:
    -------
    test_list = [my_img, my_img, my_img]

    v_concat(test_list)
    """
    if reshape:
        min_shape = sorted( [(np.sum(i.size), i.size ) for i in img_list])[0][1]
        imgs_comb = np.vstack( [np.asarray( i.resize(min_shape) ) for i in img_list] )
    else:
        imgs_comb = np.vstack(img_list)
    imgs_comb = Image.fromarray( imgs_comb)
    return imgs_comb

def vh_concat(img_list_2d, reshape = True):
    """
    contact a list "matrix" of PIL.Image images

    Arguments:
    ----------
    img_list_2d: 2d list of PIL.Image images (similar to the structure that
        would make a 2d numpy array)
    reshape: boolean, if we should reshape images to be the same size

    Returns:
    --------
        single PIL.Image image that has the original images in the matrix form

    Example:
    -------
    test_list = [[my_img, my_img, my_img],
                 [my_img, my_img, my_img]]

    vh_concat(test_list)
    """
    nrow = len(img_list_2d)
    ncol = len(img_list_2d[0])

    assert np.all([len(row) == ncol for row in img_list_2d]), \
        "img_list_2d should have matrix structure"

    if reshape:
        global_min_shape = [np.Inf, np.Inf]
        for r_idx in range(nrow):
            inner_list = img_list_2d[r_idx]
            inner_min_shape = sorted( [(np.sum(i.size), i.size ) for i in inner_list])[0][1]
            #ipdb.set_trace()
            global_min_shape[0] = np.min([global_min_shape[0], inner_min_shape[0]])
            global_min_shape[1] = np.min([global_min_shape[1], inner_min_shape[1]])

        global_list = []
        for r_idx in range(nrow):
            inner_list = img_list_2d[r_idx]
            global_list.append(
                np.hstack([np.asarray( i.resize(global_min_shape) )
                           for i in inner_list])
            )

        imgs_comb = Image.fromarray(np.vstack(global_list))

    else:
        global_list = []
        for r_idx in range(nrow):
            inner_list = img_list_2d[r_idx]
            global_list.append(np.hstack(inner_list))

        imgs_comb = Image.fromarray(np.vstack(global_list))

    return imgs_comb


def _check_dimensions(n_grobs, nrow = None, ncol = None):
    """
    Internal function to provide non-Null nrow and ncol numbers
    given a n_number of images and potentially some information about the
    desired nrow/ncols.

    Arguments:
    ----------
    n_grobs: int, number of images to be organized
    nrow: int, number of rows user wants (Default is None)
    ncol: int, number of columns user wants (Default is None)

    Returns:
    --------
        (nrow, ncol) tuple that meets user desires or errors if cannot meet
        users expectation
    """
    if nrow is None and ncol is None:
        nrow = int(np.ceil(np.sqrt(n_grobs)))
        ncol = int(np.ceil(n_grobs/nrow))
    if nrow is None:
        nrow = int(np.ceil(n_grobs/ncol))
    if ncol is None:
        ncol = int(np.ceil(n_grobs/nrow))

    assert n_grobs <= nrow * ncol, \
        "nrow * ncol < the number of grobs, please correct"

    return nrow, ncol


def _match_ids(nrow, ncol):
    """
    Internal fucntion to create the a list of images rows and column ids

    Arguments:
    ----------
    nrow: int, number of rows to have
    ncol: int, number of columns to have

    Returns:
    --------
    tuple with:
        row_id: numpy array (nrow*ncol,) with row indices
        col_id: numpy array (nrow*ncol,) with column indices

    """
    row_id = np.repeat(np.arange(nrow, dtype = int),ncol)
    col_id = np.tile(np.arange(nrow, dtype = int),ncol)

    return row_id, col_id


def arrangegrob(grobs, nrow = None, ncol = None, reshape = True,
                suppressWarnings = True):
    """
    Arranges a list of ggplot objects into a set number of rows and columns

    Arguments:
    ----------
    grobs: list of plotnine ggplot objects
    nrow: int, number of rows user wants (Default is None)
    ncol: int, number of columns user wants (Default is None)
    reshape: boolean, if we should reshape images to be the same size
    suppressWarnings: boolean, if we should suppressWarnings from ggplot
        conversion to image

    Returns:
    --------
    single PIL.Image image that has the original images in the matrix form


    Details:
    --------
    Similar to a basic version of R library gridExtra::arrangeGrob function

    """
    # initial setup -----------

    n_grobs = len(grobs)
    nrow, ncol = _check_dimensions(n_grobs, nrow, ncol)
    row_id, _ = _match_ids(nrow, ncol)

    if n_grobs < nrow * ncol:
        n_diff = nrow * ncol - n_grobs
        grobs = grobs + [p9.ggplot(pd.DataFrame()) + p9.theme_void() for _ in range(n_diff)]

    # converting to images -----------
    if suppressWarnings:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            img_list = [gg2img(gg) for gg in grobs]
    else:
        img_list = [gg2img(gg) for gg in grobs]
    # reshape if necessary ----------

    if reshape:
        min_shape = sorted( [(np.sum(i.size), i.size ) for i in img_list])[0][1]
        img_list = [i.resize(min_shape)  for i in img_list]

    # combining

    global_list = []
    for r_idx in range(nrow):
        inner_img_id = np.ix_(row_id == r_idx)[0]
        inner_img_list = [img_list[good_i] for good_i in inner_img_id]
        global_list.append(np.hstack(inner_img_list))


    single_image = Image.fromarray(np.vstack(global_list))

    return single_image
