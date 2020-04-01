# This is a test of reading vedio
from UpdateFuncs import *
from get_LBP_from_Image import *
import os
from LBPfuncs import CompareLBP

sysbg = cv2.createBackgroundSubtractorMOG2(500, 30, detectShadows=True)
# sysKNN = cv2.createBackgroundSubtractorKNN(500, 50, detectShadows=True)
"""文件的读取与视频文件的初始化"""
c_path = os.getcwd()
v_path = "D:\\Programming\\MATLAB\\video_prog\\"
methodInPath = r"\System"
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
UpdateThred = 20
DelayWaitFrameNum = 60
NumFrameForceForeToBack = 100  # 当一个目标多少帧之后，它会强制转换为背景
kernel1 = np.ones((3, 3), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)
kernel3 = np.ones((3, 3), np.uint8)
kernel4 = np.ones((9, 9), np.uint8)
chn = 3  # 色彩通道数
shape = (np.int(1080 * 0.4), np.int(720 * 0.4), chn)
size = (np.int(720 * 0.4), np.int(1080 * 0.4))
"""过程中承载参数的矩阵的设置"""
BG = np.zeros(shape=shape, dtype=np.uint8)
Sub = np.ones(shape=shape, dtype=np.uint8)
ColorSubShow = np.ones(shape=shape, dtype=np.uint8)
SubR = np.ones(shape=shape, dtype=np.uint8)
SubTemp = np.zeros(shape=shape, dtype=np.uint8)
ForeFlag = np.zeros(shape=shape, dtype=np.int32)
FlagOld = np.zeros(shape=shape, dtype=np.int32)
LongNotGrowing = np.zeros(shape=shape, dtype=np.int32)
DelayFlag = np.zeros(shape=shape, dtype=np.int32)
SubInInt32 = np.zeros(shape=shape, dtype=np.int32)
BG_DiffFlag = np.zeros(shape=shape, dtype=np.uint8)
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
"""背景初始化，采用第一帧初始化"""
ret, initialframe = vid.read()
initialframe = cv2.resize(initialframe, size, interpolation=cv2.INTER_CUBIC)
FrameNum = 0  # 当前帧计数器
BG = initialframe
"""各种更新时的策略选择"""
updateAsAll = True  # 按照红绿蓝三色更新
UpdateSeparately = True  # 是否将前后背景分开更新
EliminateForegroundTooLong = True  # 定期消除长时间占用前景像素
UseMinimumRecContours = True  # 使用最小矩形框选选择目标
UpdateWithinContours = True  # 以轮廓内物体为前景更新
DoMorphology_1 = True  # 使用形态学处理消除小区域，先开后闭
DoMorphology_2 = True  # 获得轮廓后使用形态学处理消除小区域，先开后闭
GenContours = True  # 显示物体轮廓
MedBlur = False  # 中值滤波选择
"""策略选择自动纠错环节"""
CheckTackle(GenContours, UseMinimumRecContours, UpdateWithinContours, UpdateSeparately, a, b)
lbp = LBP()
"""视频目标追踪循环开始"""
while 1:
    FrameNum += 1
    ret, frame_org = vid.read()
    if not ret:
        print("程序出错，没有正确读取视频，报错点在刚进入循环")
        break
    frame_org = cv2.resize(frame_org, size, interpolation=cv2.INTER_CUBIC)
    frame = frame_org[:, :, 0:chn]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = CompareLBP(gray, BG, windowSize=32, step=8)
    cv2.imshow("LBP",out)

    # lbphist = lbp.lbp_basic(gray)
    # cv2.imshow("LBP", lbphist)
    # lbp.show_basic_hist(lbphist)
    plt.show()
    SystemSub = sysbg.apply(frame)
    cv2.imshow("System", SystemSub)
    Sub1 = cv2.absdiff(BG, frame)
    ret, Sub = cv2.threshold(Sub1, BinaryThreshold, 255, type=cv2.THRESH_BINARY)
    """是否清除过长的前景"""
    if FrameNum > 0:
        if EliminateForegroundTooLong:
            EliminateForeGround(Sub, ForeFlag, FlagOld, NumFrameForceForeToBack, LongNotGrowing, DelayFlag,
                                DelayWaitFrameNum)
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
        contours, hierarchy = cv2.findContours(image=SubAll, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
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
    """显示各个图像"""
    cv2.imshow('Current Frame (Colored)', frame)
    cv2.imshow("Current Background (Colored)", BG)
    cv2.imshow("Sub (Colored)", ColorSubShow)
    cv2.imshow("Sub with All color channel merged", SubAll)
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
        cv2.imwrite(pathForFrameFore, SystemSub)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
