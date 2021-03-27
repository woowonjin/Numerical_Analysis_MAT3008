import cv2
import numpy as np


def correlation(img):
    B, G, R = cv2.split(img)
    B_f = B.flatten()
    G_f = G.flatten()
    R_f = R.flatten()
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(img_yuv)
    Y_f = Y.flatten()
    U_f = U.flatten()
    V_f = V.flatten()
    # print("B-G")
    # print(np.corrcoef(B_f, G_f)[0][1])
    # print("G-R")
    # print(np.corrcoef(G_f, R_f)[0][1])
    # print("R-B")
    # print(np.corrcoef(R_f, B_f)[0][1])
    # print("Y-U")
    # print(np.corrcoef(Y_f, U_f)[0][1])
    # print("U-V")
    # print(np.corrcoef(U_f, V_f)[0][1])
    # print("V-Y")
    # print(np.corrcoef(V_f, Y_f)[0][1])
    print("Y-G")
    print(np.corrcoef(Y_f, G_f)[0][1])


for i in range(1, 11):
    print("img : ", str(i))
    img = cv2.imread(str(i) + ".jpg", cv2.IMREAD_COLOR)
    # img = cv2.resize(img, dsize=(512, 512),
    #                  interpolation=cv2.INTER_AREA)
    correlation(img)
