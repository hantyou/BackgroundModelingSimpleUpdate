# This is a test of reading vedio

from os import getcwd
import os
from multiprocessing import Pool
from threading import Thread
from multiprocessing import Process

from LBPfuncs import CompareLBP, MyResize
from UpdateFuncs import *
from get_LBP_from_Image import *
import matplotlib.pyplot as plt
from PIL import Image
import torch

print(torch.__version__)

from run_placesCNN_unified import placeCNN

sysbg = cv2.createBackgroundSubtractorMOG2(500, 30, detectShadows=True)
# sysKNN = cv2.createBackgroundSubtractorKNN(500, 50, detectShadows=True)
"""文件的读取与视频文件的初始化"""
c_path = getcwd()
v_path = "D:\\Programming\\MATLAB\\video_prog\\"
methodInPath = r"\tpres"
v_filename = "FroggerHighway.mp4"
filepath = v_path + "\\" + v_filename
vid = cv2.VideoCapture(filepath)
CheckVideo(vid)
h = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
w = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
"""各类参数与过程中用到的计算用矩阵"""
a = 0.055  # 更新率，a为背景更新率，b为前景更新率
b = 0.001
BinaryThreshold = 30
LBP_threshold = 3
decay = 0.0000001
UpdateThred = 20
DelayWaitFrameNum = 80
NumFrameForceForeToBack = 100  # 当一个目标多少帧之后，它会强制转换为背景
kernel1 = np.ones((1, 1), np.uint8)
kernel2 = np.ones((2, 2), np.uint8)
kernel3 = np.ones((3, 3), np.uint8)
kernel4 = np.ones((3, 3), np.uint8)
chn = 3  # 色彩通道数
w = 256
h = 256
shapewh = (np.int(w), np.int(h), chn)
shapehw = (np.int(h), np.int(w), chn)
sizewh = (np.int(w), np.int(h))
sizehw = (np.int(h), np.int(w))
"""过程中承载参数的矩阵的设置"""
BG = np.zeros(shape=shapehw, dtype=np.uint8)
Sub = np.ones(shape=shapehw, dtype=np.uint8)
SubAll = np.zeros(shape=sizehw, dtype=np.uint8)
ColorSubShow = np.ones(shape=shapehw, dtype=np.uint8)
SubR = np.ones(shape=shapehw, dtype=np.uint8)
SubTemp = np.zeros(shape=shapehw, dtype=np.uint8)
ForeFlag = np.zeros(shape=shapehw, dtype=np.int32)
FlagOld = np.zeros(shape=shapehw, dtype=np.int32)
LongNotGrowing = np.zeros(shape=shapehw, dtype=np.int32)
DelayFlag = np.zeros(shape=shapehw, dtype=np.int32)
SubInInt32 = np.zeros(shape=shapehw, dtype=np.int32)
BG_DiffFlag = np.zeros(shape=shapehw, dtype=np.uint8)
"""显示窗口初始化"""
cv2.namedWindow("Current Frame (Colored)", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Current Background (Colored)", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Sub (Colored)", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Sub with All color channel merged", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Region Flaged as Backround in Current Frame", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Region Flaged as Foreground in Current Frame", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("contours", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("System", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("ForeFlag", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("LBP", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("LBP Fore", cv2.WINDOW_KEEPRATIO)
"""背景初始化，采用第一帧初始化"""
ret, initialframe = vid.read()
initialframe = cv2.resize(initialframe, sizewh, interpolation=cv2.INTER_CUBIC)
FrameNum = 0  # 当前帧计数器
BG = initialframe
"""各种更新时的策略选择"""
updateAsAll = True  # 按照红绿蓝三色更新
UpdateSeparately = True  # 是否将前后背景分开更新
EliminateForegroundTooLong = True  # 定期消除长时间占用前景像素
UseMinimumRecContours = False  # 使用最小矩形框选选择目标
UpdateWithinContours = True  # 以轮廓内物体为前景更新
DoMorphology_1 = True  # 使用形态学处理消除小区域，先开后闭
DoMorphology_2 = False  # 获得轮廓后使用形态学处理消除小区域，先开后闭
GenContours = True  # 显示物体轮廓
MedBlur = False  # 中值滤波选择
UseLBP = False  # 使用LBP纹理信息
"""策略选择自动纠错环节"""
CheckTackle(GenContours, UseMinimumRecContours, UpdateWithinContours, UpdateSeparately, a, b)
lbp = LBP()


def showpics():
    plt.figure(1)
    # pltFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pltFrame = frame
    plt.subplot(221)
    plt.imshow(pltFrame)
    plt.title('Current Frame (Colored)', fontsize=8)
    plt.xticks([])
    plt.yticks([])
    # pltBG = cv2.cvtColor(BG, cv2.COLOR_BGR2RGB)
    pltBG = BG
    plt.subplot(222)
    plt.imshow(pltBG)
    plt.title('Current Background (Colored)', fontsize=8)
    plt.xticks([])
    plt.yticks([])
    # pltColorSubShow = cv2.cvtColor(ColorSubShow, cv2.COLOR_BGR2RGB)
    pltColorSubShow = ColorSubShow
    plt.subplot(223)
    plt.imshow(pltColorSubShow)
    plt.title('Sub (Colored)', fontsize=8)
    plt.xticks([])
    plt.yticks([])
    # pltSubAll = cv2.cvtColor(SubAll, cv2.COLOR_BGR2RGB)
    pltSubAll = SubAll
    plt.subplot(224)
    plt.imshow(pltSubAll)
    plt.title('Sub with All color channel merged', fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.pause(0.00001)
"""视频目标追踪循环开始"""
if __name__ == '__main__':
    plt.ion()
    while 1:
        FrameNum += 1
        ret, frame_org = vid.read()
        if not ret:
            print("程序出错，没有正确读取视频，报错点在刚进入循环")
            break
        frame_org = cv2.resize(frame_org, sizewh, interpolation=cv2.INTER_CUBIC)
        frame = frame_org[:, :, 0:chn]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayBG = cv2.cvtColor(BG, cv2.COLOR_BGR2GRAY)
        # ShowSubIm("Current Frame and BG", frame_org, grayBG)

        # lbphist = lbp.lbp_basic(gray)
        # cv2.imshow("LBP", lbphist)
        # lbp.show_basic_hist(lbphist)
        # plt.show()
        SystemSub = sysbg.apply(frame)
        cv2.imshow("System", SystemSub)
        Sub1 = cv2.absdiff(BG, frame)
        ret, Sub = cv2.threshold(Sub1, BinaryThreshold, 255, type=cv2.THRESH_BINARY)
        """是否清除过长的前景"""
        if FrameNum > 0:
            if EliminateForegroundTooLong:
                SubInInt32 = np.where(Sub.copy() < 1, 0, 1)
                ForeFlag = ForeFlag + SubInInt32
                NotGrowing = np.where(FlagOld == ForeFlag, 0, 1)
                InvNotGrowing = np.where(FlagOld == ForeFlag, 1, 0)
                NotGrowing = np.where(ForeFlag >= NumFrameForceForeToBack, 1, NotGrowing)
                LongNotGrowing = LongNotGrowing + InvNotGrowing
                ForeFlag = ForeFlag * np.where(LongNotGrowing > 20, 0, 1)
                LongNotGrowing = np.where(LongNotGrowing > 20, 0, LongNotGrowing)
                Sub = np.where(LongNotGrowing <= 20, Sub, 0)
                Sub = np.where(ForeFlag < NumFrameForceForeToBack, Sub, 0)  # 清除存在超过NumFrameForceForeToBack帧的前景
                DelayFlag = np.where(ForeFlag >= NumFrameForceForeToBack, DelayFlag + 1, DelayFlag)
                ForeFlag = np.where(DelayFlag > DelayWaitFrameNum, 0, ForeFlag)
                DelayFlag = np.where(DelayFlag > DelayWaitFrameNum, 0, DelayFlag)
                cv2.imshow("ForeFlag", np.uint8(ForeFlag))
                FlagOld = ForeFlag
        Old_BG = BG.copy()
        """是否做形态学处理"""
        if DoMorphology_1:
            Sub = cv2.morphologyEx(Sub, cv2.MORPH_OPEN, kernel1)  # 开
            Sub = cv2.morphologyEx(Sub, cv2.MORPH_CLOSE, kernel2)  # 闭
        ColorSubShow = Sub.copy()
        SubAll = Sub[:, :, 0] + Sub[:, :, 1] + Sub[:, :, 2]  # 统一三个色彩通道的前景探测结果
        SubAll = np.where(SubAll < 1, SubAll, 255)
        """选择是否做中值滤波消除噪声干扰"""
        if MedBlur:
            Sub = cv2.medianBlur(Sub, 3)
        """生成轮廓"""
        if GenContours:
            # contours, hierarchy = cv2.findContours(image=SubAll, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            contours = cv2.findContours(image=SubAll, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[0]
            FrameForContours = frame.copy()
            if UseMinimumRecContours:  # 生成最小矩阵轮廓
                SubTemp = np.where(SubTemp == 0, SubTemp, 0)
                for i, contour in enumerate(contours):
                    rct = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rct)
                    box = np.int0(box)
                    cv2.drawContours(FrameForContours, [box], 0, (255, 255, 255), -1)
                    cv2.drawContours(SubTemp, [box], 0, (255, 255, 255), -1)
            else:  # 使用不规则轮廓
                SubTemp = np.where(SubTemp == 0, SubTemp, 0)
                for i, contour in enumerate(contours):
                    cv2.drawContours(FrameForContours, contours, i, (255, 255, 255), -1)
                    cv2.drawContours(SubTemp, contours, i, (255, 255, 255), -1)
            cv2.imshow("contours", FrameForContours)
            """判断是否使用轮廓内的内容进行升级"""
            if UpdateWithinContours:
                Sub = np.where(SubTemp < 1, Sub, 255)
                if DoMorphology_2:
                    Sub = cv2.morphologyEx(Sub, cv2.MORPH_OPEN, kernel3)  # 开
                    Sub = cv2.morphologyEx(Sub, cv2.MORPH_CLOSE, kernel4)  # 闭
        """如果不使用轮廓升级"""
        if not UpdateWithinContours:  # 不使用轮廓
            """判断是否使用全部色彩层信息进行背景升级"""
            if updateAsAll:
                for i in range(chn):
                    Sub[:, :, i] = SubAll
        """生成前景背景掩膜处理后的"""

        SubR = np.uint8(Sub / 255)  # SubR用来做掩模版，区分出前景和后景
        # out = CompareLBP(cv2.pyrDown(gray), cv2.pyrDown(grayBG), cv2.pyrDown(SubAll / 255), windowSize=8, step=8)

        if not FrameNum % 1:
            if UseLBP:
                out = CompareLBP(MyResize(gray, 1.5), MyResize(grayBG, 1.5), MyResize(SubAll / 255, 1.5), windowSize=10,
                                 step=10, region_thresh=2, decay=decay)

                out = cv2.GaussianBlur(out, (0, 0), sigmaX=1.1,
                                       sigmaY=1.1)
                cv2.imshow("LBP", out)
                out = cv2.resize(out, (SubR.shape[0], SubR.shape[1]))
                ret, out = cv2.threshold(out, LBP_threshold, 1, type=cv2.THRESH_BINARY)
                for i in range(chn):
                    SubR[:, :, i] = SubR[:, :, i] + out
                SubR = np.uint8(np.where(SubR > 0, 1, 0))
                cv2.imshow("LBP Fore", np.uint8(out * gray))
        BG_ForeD = BG * SubR
        BG_BackD = BG - BG_ForeD
        # cv2.imshow("Region Flaged as Background in Current Background", BG_BackD[:, :, 0])
        # cv2.imshow("Region Flaged as Foreground in Current Background", BG_ForeD[:, :, 0])
        frame_fore = frame * SubR
        frame_back = frame - frame_fore
        cv2.imshow("Region Flaged as Backround in Current Frame", frame_back[:, :, 0])
        cv2.imshow("Region Flaged as Foreground in Current Frame", frame_fore[:, :, :])
        BG_Back = cv2.addWeighted(BG_BackD, 1 - a, frame_back, a, 0)
        BG_Fore = cv2.addWeighted(BG_ForeD, 1 - b, frame_fore, b, 0)
        BG = BG_Back + BG_Fore
        """控制背景更新率"""
        BG_Diff = cv2.absdiff(Old_BG, BG)
        BG_DiffFlag = np.where(BG_Diff < UpdateThred, BG_DiffFlag, 1)
        Old_BG_Reserve = Old_BG * BG_DiffFlag
        BG_NotUpdate = BG * BG_DiffFlag
        BG_Update = BG - BG_NotUpdate
        BG = BG_Update + Old_BG_Reserve
        """To the CNN"""
        if FrameNum % 10 is 0:
            i = os.system('cls')
            placeCNN(BG, FrameNum,'BG-')
            placeCNN(frame_org, FrameNum,'Frame-')
        """显示各个图像"""
        if FrameNum ==1:
            s = Process(target=showpics)
        s.run()
        #cv2.imshow('Current Frame (Colored)', frame)
        #cv2.imshow("Current Background (Colored)", BG)
        #cv2.imshow("Sub (Colored)", ColorSubShow)
        #cv2.imshow("Sub with All color channel merged", SubAll)
        """Renew Paths"""
        if not FrameNum % 10:
            pathForSubAll = c_path + r"\Results" + methodInPath + "\\" + v_filename + r"\SubAll\SubAll-%d" % FrameNum + ".jpg"
            pathForRealtimeBackground = c_path + r"\Results" + methodInPath + "\\" + \
                                        v_filename + r"\RealtimeBackground\RealtimeBackground-%d" % FrameNum + ".jpg"
            pathForFrame = c_path + r"\Results" + methodInPath + "\\" + v_filename + r"\Frame\Frame-%d" % FrameNum + ".jpg"
            pathForColorSub = c_path + r"\Results" + methodInPath + "\\" + \
                              v_filename + r"\ColorSub\ColorSub-%d" % FrameNum + ".jpg"
            if EliminateForegroundTooLong:
                pathForForeFlag = c_path + r"\Results" + methodInPath + "\\" + \
                                  v_filename + r"\ForeFlag\ForeFlag-%d" % FrameNum + ".jpg"
                cv2.imwrite(pathForForeFlag, ForeFlag)
            pathForFrameFore = c_path + r"\Results" + methodInPath + "\\" + v_filename + r"\FrameFore\FrameFore-%d" % FrameNum + ".jpg"
            cv2.imwrite(pathForSubAll, SubAll)
            cv2.imwrite(pathForRealtimeBackground, BG)
            cv2.imwrite(pathForFrame, frame)
            cv2.imwrite(pathForColorSub, ColorSubShow)
            cv2.imwrite(pathForFrameFore, frame_fore)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    plt.ioff()
    plt.show()
    vid.release()
    cv2.destroyAllWindows()
