import cv2
import numpy as np

import get_LBP_from_Image as LBP


def region_flag(Sub, yt, xt, windowSize, region_thresh):
    return np.sum(Sub[yt:yt + windowSize, xt:xt + windowSize]) > (windowSize ** 2) / region_thresh


def region_flag_out(out, yt, xt, windowSize, region_thresh_out):
    return np.sum(out[yt:yt + windowSize, xt:xt + windowSize]) <= region_thresh_out


def MyResize(I, factor):
    w = I.shape[0]
    h = I.shape[1]
    w0 = int(w / factor)
    h0 = int(h / factor)
    out = cv2.resize(I, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return out


def calc_regional_LBP(target, background, lbp, out, yt, xt, windowSize):
    t_windowed = target[yt:yt + windowSize, xt:xt + windowSize]
    b_windowed = background[yt:yt + windowSize, xt:xt + windowSize]
    t_LBP_map = lbp.lbp_uniform(t_windowed)
    b_LBP_map = lbp.lbp_uniform(b_windowed)
    t_hist = lbp.get_uniform_hist(t_LBP_map)
    b_hist = lbp.get_uniform_hist(b_LBP_map)
    d = LBP.chi2_distance(t_hist, b_hist)
    # print((j, i))
    out[yt:yt + windowSize, xt:xt + windowSize] += np.ones((windowSize, windowSize)) * d


def CompareLBP(target, background, Sub, windowSize=32, step=4, region_thresh=3, decay=0.05):
    print("Calculating LBP of Image")
    [h, w] = [target.shape[0], target.shape[1]]
    x_iter = int((w - windowSize) / step) + 1
    y_iter = int((h - windowSize) / step) + 1
    x_left = w - windowSize - step * (x_iter - 1)
    y_left = h - windowSize - step * (y_iter - 1)
    out = np.zeros((h, w))
    lbp = LBP.LBP()
    flag = 0
    deviation = decay
    deviation1 = deviation * 0.75
    deviation2 = deviation * 0.75
    deviation3 = deviation * 0.75
    deviation4 = deviation * 0.75
    for j in range(y_iter):
        for i in range(x_iter):
            xt = i * step
            yt = j * step
            CalcFlag = (np.sum(Sub[yt:yt + windowSize, xt:xt + windowSize]) > (windowSize ** 2) / region_thresh)
            if j < y_iter - 2 and j > 1 and i < x_iter - 2 and i > 1:
                CalcFlag = CalcFlag or region_flag(Sub, yt + windowSize, xt - windowSize, windowSize, region_thresh)
                CalcFlag = CalcFlag or region_flag(Sub, yt + windowSize, xt, windowSize, region_thresh)
                CalcFlag = CalcFlag or region_flag(Sub, yt + windowSize, xt + windowSize, windowSize, region_thresh)
                CalcFlag = CalcFlag or region_flag(Sub, yt, xt + windowSize, windowSize, region_thresh)
                CalcFlag = CalcFlag and region_flag_out(out, yt + windowSize, xt - windowSize, windowSize, 1)
                CalcFlag = CalcFlag and region_flag_out(out, yt + windowSize, xt, windowSize, 1)
                CalcFlag = CalcFlag and region_flag_out(out, yt + windowSize, xt + windowSize, windowSize, 1)
                CalcFlag = CalcFlag and region_flag_out(out, yt, xt + windowSize, windowSize, 1)
            if CalcFlag:
                flag += 3
                if flag > 0:
                    calc_regional_LBP(target, background, lbp, out, yt, xt, windowSize)

                    if j < y_iter - 2 and j > 1 and i < x_iter - 2 and i > 1:
                        # if not region_flag(Sub, yt - windowSize, xt - windowSize, windowSize, region_thresh):
                        if region_flag(out, yt - windowSize, xt - windowSize, windowSize, region_thresh * deviation1):
                            calc_regional_LBP(target, background, lbp, out, yt - windowSize, xt - windowSize,
                                              windowSize)
                        if region_flag(out, yt - windowSize, xt, windowSize, region_thresh * deviation2):
                            calc_regional_LBP(target, background, lbp, out, yt - windowSize, xt, windowSize)
                        if region_flag(out, yt - windowSize, xt + windowSize, windowSize, region_thresh * deviation3):
                            calc_regional_LBP(target, background, lbp, out, yt - windowSize, xt + windowSize,
                                              windowSize)
                        if region_flag(out, yt, xt - windowSize, windowSize, region_thresh * deviation4):
                            calc_regional_LBP(target, background, lbp, out, yt, xt - windowSize, windowSize)
                        if region_flag(out, yt, xt + windowSize, windowSize, region_thresh * deviation):
                            calc_regional_LBP(target, background, lbp, out, yt, xt + windowSize, windowSize)
                        if region_flag(out, yt + windowSize, xt - windowSize, windowSize, region_thresh * deviation):
                            calc_regional_LBP(target, background, lbp, out, yt + windowSize, xt - windowSize,
                                              windowSize)
                        if region_flag(out, yt + windowSize, xt, windowSize, region_thresh * deviation):
                            calc_regional_LBP(target, background, lbp, out, yt + windowSize, xt, windowSize)
                        if region_flag(out, yt + windowSize, xt + windowSize, windowSize, region_thresh * deviation):
                            calc_regional_LBP(target, background, lbp, out, yt + windowSize, xt + windowSize,
                                              windowSize)

                    flag -= 1
                # LBP.ShowSubIm("out", out, out[yt:yt + windowSize, xt:xt + windowSize])

                # plt.suptitle("out")
        """
        plt.subplot(1, 2, 1)
        plt.imshow(out)
        plt.subplot(1, 2, 2)
        plt.imshow(out)
        plt.pause(0.00001)
        """

    if x_left:
        for j in range(y_iter):
            yt = j * step
            t_windowed = target[yt:yt + windowSize, w - x_left:w]
            b_windowed = background[yt:yt + windowSize, w - x_left:w]
            t_LBP_map = lbp.lbp_revolve(t_windowed)
            b_LBP_map = lbp.lbp_revolve(b_windowed)
            t_hist = lbp.get_revolve_hist(t_LBP_map)
            b_hist = lbp.get_revolve_hist(b_LBP_map)
            d = LBP.chi2_distance(t_hist, b_hist)
            out[yt:yt + windowSize, w - x_left:w] += np.ones((windowSize, x_left)) * d
    if y_left:
        for i in range(x_iter):
            xt = i * step
            t_windowed = target[h - y_left:h, xt:xt + windowSize]
            b_windowed = background[h - y_left:h, xt:xt + windowSize]
            t_LBP_map = lbp.lbp_revolve(t_windowed)
            b_LBP_map = lbp.lbp_revolve(b_windowed)
            t_hist = lbp.get_revolve_hist(t_LBP_map)
            b_hist = lbp.get_revolve_hist(b_LBP_map)
            d = LBP.chi2_distance(t_hist, b_hist)
            out[h - y_left:h, xt:xt + windowSize] += np.ones((y_left, windowSize)) * d
    if y_left and x_left:
        t_windowed = target[h - y_left:h, w - x_left:w]
        b_windowed = background[h - y_left:h, w - x_left:w]
        t_LBP_map = lbp.lbp_revolve(t_windowed)
        b_LBP_map = lbp.lbp_revolve(b_windowed)
        t_hist = lbp.get_revolve_hist(t_LBP_map)
        b_hist = lbp.get_revolve_hist(b_LBP_map)
        d = LBP.chi2_distance(t_hist, b_hist)
        out[h - y_left:h, w - x_left:w] += np.ones((y_left, x_left)) * d
    # out = out / np.max(out)
    return out
