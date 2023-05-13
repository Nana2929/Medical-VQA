def hu_transform(img, ww, wl, dst_range=(0, 1)):
    """
    Apply Hounsfield Unit (HU) transformation to the image.
    :param img: input image
    :param ww: window width
    :param wl: window level
    :param dst_range: output range
    :return: transformed image
    """
    src_min = wl - ww / 2
    src_max = wl + ww / 2

    outputs = (img - src_min) / ww * (dst_range[1] -
                                      dst_range[0]) + dst_range[0]
    outputs[img >= src_max] = 1
    outputs[img <= src_min] = 0

    return outputs * 255