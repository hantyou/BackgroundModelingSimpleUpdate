import numpy as np

import get_LBP_from_Image as LBP


def CompareLBP(target, background, windowSize=32, step=1):
    [h, w] = [target.shape[0], target.shape[1]]
    x_iter = int((w - windowSize) / step)
    y_iter = int((h - windowSize) / step)
    x_left = w - windowSize - step * x_iter
    y_left = h - windowSize - step * y_iter
    out = np.zeros((h, w))
    lbp = LBP.LBP()
    for j in range(y_iter):
        for i in range(x_iter):
            xt = i * step
            yt = j * step
            t_windowed = target[yt:yt + windowSize, xt:xt + windowSize]
            b_windowed = background[yt:yt + windowSize, xt:xt + windowSize]
            t_LBP_map = lbp.lbp_revolve(t_windowed)
            b_LBP_map = lbp.lbp_revolve(b_windowed)
            t_hist = lbp.get_revolve_hist(t_LBP_map)
            b_hist = lbp.get_revolve_hist(b_LBP_map)
            d = LBP.chi2_distance(t_hist, b_hist)
            out[yt:yt + windowSize, xt:xt + windowSize] += np.ones((windowSize, windowSize)) * d
    if x_left:
        for j in range(y_iter):
            yt = j * step
            t_windowed = target[yt:yt + windowSize, w - windowSize:w]
            b_windowed = background[yt:yt + windowSize, w - windowSize:w]
            t_LBP_map = lbp.lbp_revolve(t_windowed)
            b_LBP_map = lbp.lbp_revolve(b_windowed)
            t_hist = lbp.get_revolve_hist(t_LBP_map)
            b_hist = lbp.get_revolve_hist(b_LBP_map)
            d = LBP.chi2_distance(t_hist, b_hist)
            out[yt:yt + windowSize, w - windowSize:w] += np.ones((windowSize, windowSize)) * d
    if y_left:
        for i in range(x_iter):
            xt = i * step
            t_windowed = target[h - windowSize:h, xt:xt + windowSize]
            b_windowed = background[h - windowSize:h, xt:xt + windowSize]
            t_LBP_map = lbp.lbp_revolve(t_windowed)
            b_LBP_map = lbp.lbp_revolve(b_windowed)
            t_hist = lbp.get_revolve_hist(t_LBP_map)
            b_hist = lbp.get_revolve_hist(b_LBP_map)
            d = LBP.chi2_distance(t_hist, b_hist)
            out[h - windowSize:h, xt:xt + windowSize] += np.ones((windowSize, windowSize)) * d
    if y_left and x_left:
        t_windowed = target[h - windowSize:h, w - windowSize:w]
        b_windowed = background[h - windowSize:h, w - windowSize:w]
        t_LBP_map = lbp.lbp_revolve(t_windowed)
        b_LBP_map = lbp.lbp_revolve(b_windowed)
        t_hist = lbp.get_revolve_hist(t_LBP_map)
        b_hist = lbp.get_revolve_hist(b_LBP_map)
        d = LBP.chi2_distance(t_hist, b_hist)
        out[h - windowSize:h, w - windowSize:w] += np.ones((windowSize, windowSize)) * d
    out = out / np.max(out)
    return out
