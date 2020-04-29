import cv2
import matplotlib.pyplot as plt


def MyImshow(I):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    plt.imshow(I)
    plt.show()


def ShowSubplot(I, B):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 2, 1)
    plt.imshow(I)
    plt.subplot(1, 2, 2)
    plt.imshow(B)
    plt.show()


BG = cv2.imread("RealtimeBackground-10.jpg")
I300 = cv2.imread("Frame-300.jpg")
I530 = cv2.imread("Frame-530.jpg")
I680 = cv2.imread("Frame-680.jpg")
B300 = cv2.imread("SubAll-300.jpg")
B530 = cv2.imread("SubAll-530.jpg")
B680 = cv2.imread("SubAll-680.jpg")
MyImshow(BG)
ShowSubplot(I300, B300)
ShowSubplot(I530, B530)
ShowSubplot(I680, B680)
