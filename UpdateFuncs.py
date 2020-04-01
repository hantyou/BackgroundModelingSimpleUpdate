import cv2
import numpy as np


def CheckVideo(vid):
    flag = vid.isOpened()
    if flag:
        print("打开摄像头成功")
    else:
        print("打开摄像头失败")


def EliminateForeGround(Sub, ForeFlag, FlagOld, NumFrameForceForeToBack, LongNotGrowing, DelayFlag, DelayWaitFrameNum):
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


def CheckTackle(GenContours, UseMinimumRecContours, UpdateWithinContours, UpdateSeparately, a, b):
    if not GenContours:
        if UseMinimumRecContours:
            UseMinimumRecContours = False
            print("没有生成轮廓，以最小矩形轮廓更新被自动取消")
        if UpdateWithinContours:
            UpdateWithinContours = False
            print("没有生成轮廓，以轮廓更新被自动取消")
    else:
        if UseMinimumRecContours:
            if not UpdateWithinContours:
                print("虽然最小矩形轮廓被标出，但更新背景时并没有用到")
        else:
            if not UpdateWithinContours:
                print("虽然不规则轮廓被标出，但更新背景时并没有用到")
    if not UpdateSeparately:
        b = a
