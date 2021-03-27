import cv2
import numpy as np
from math import cos, pi, sqrt


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    for i in range(len(flat)):
        flat[i] = abs(flat[i])
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def dct(img):
    (img_B, img_G, img_R) = cv2.split(img)
    result_B = np.zeros((128, 128), dtype=np.float, order="C")
    result_G = np.zeros((128, 128), dtype=np.float, order="C")
    result_R = np.zeros((128, 128), dtype=np.float, order="C")
    for i in range(0, 8):  # 전체에서 block row
        for j in range(0, 8):  # 전체에서 block column
            F_B = [[0] * 16] * 16
            F_G = [[0] * 16] * 16
            F_R = [[0] * 16] * 16
            for u in range(0, 16):  # block에서 row
                c_u = 1
                if u == 0:
                    c_u = sqrt(1 / 2)
                for v in range(0, 16):  # block에서 column
                    c_v = 1
                    if v == 0:
                        c_v = sqrt(1 / 2)
                    F_vu_B = 0
                    F_vu_G = 0
                    F_vu_R = 0
                    for x in range(0, 16):
                        x_cos = cos(u * pi * (2 * x + 1) / 32)
                        y_arr = []
                        for y in range(0, 16):
                            y_cos = cos(v * pi * (2 * y + 1) / 32)
                            y_arr.append(y_cos)
                            s_yx_B = img_B[i * 16 + x][j * 16 + y]
                            s_yx_G = img_G[i * 16 + x][j * 16 + y]
                            s_yx_R = img_R[i * 16 + x][j * 16 + y]
                            F_vu_B += s_yx_B * y_cos * x_cos
                            F_vu_G += s_yx_G * y_cos * x_cos
                            F_vu_R += s_yx_R * y_cos * x_cos
                        # if u == 0 and v == 1:
                        # print("Y_ARR : ", y_arr)
                    F_B[u][v] = F_vu_B * c_v * c_u / 8
                    F_G[u][v] = F_vu_G * c_v * c_u / 8
                    F_R[u][v] = F_vu_R * c_v * c_u / 8
                    # if i == 0 and j == 0:
                    # print("F : ", F_B[u][0])
            # if i == 0 and j == 0:
            # print(F_B)
            arr_B = []
            arr_G = []
            arr_R = []
            s_B, v_B = largest_indices(np.array(F_B), 16)
            s_G, v_G = largest_indices(np.array(F_G), 16)
            s_R, v_R = largest_indices(np.array(F_R), 16)
            for l in range(16):
                arr_B.append((s_B[l], v_B[l]))
                arr_G.append((s_G[l], v_G[l]))
                arr_R.append((s_R[l], v_R[l]))
            for a in range(16):
                for b in range(16):
                    if (a, b) not in arr_B:
                        F_B[a][b] = 0
                    if (a, b) not in arr_G:
                        F_G[a][b] = 0
                    if (a, b) not in arr_R:
                        F_R[a][b] = 0
            for x in range(16):
                for y in range(16):
                    r_B = 0
                    r_G = 0
                    r_R = 0
                    for u in range(16):
                        c_u = 1
                        if u == 0:
                            c_u = sqrt(1 / 2)
                        for v in range(16):
                            c_v = 1
                            if v == 0:
                                c_v = sqrt(1 / 2)
                            r_B += (
                                c_u
                                * c_v
                                * F_B[u][v]
                                * cos(v * pi * (2 * y + 1) / 32)
                                * cos(u * pi * (2 * x + 1) / 32)
                            )
                            r_G += (
                                c_u
                                * c_v
                                * F_G[u][v]
                                * cos(v * pi * (2 * y + 1) / 32)
                                * cos(u * pi * (2 * x + 1) / 32)
                            )
                            r_R += (
                                c_u
                                * c_v
                                * F_R[u][v]
                                * cos(v * pi * (2 * y + 1) / 32)
                                * cos(u * pi * (2 * x + 1) / 32)
                            )
                    result_B[i * 16 + x][j * 16 + y] = r_B / 8
                    result_G[i * 16 + x][j * 16 + y] = r_G / 8
                    result_R[i * 16 + x][j * 16 + y] = r_R / 8
    for i in range(128):
        for j in range(128):
            if result_B[i][j] < 0:
                result_B[i][j] = 0
            if result_B[i][j] > 255:
                result_B[i][j] = 255
            if result_G[i][j] < 0:
                result_G[i][j] = 0
            if result_G[i][j] > 255:
                result_G[i][j] = 255
            if result_R[i][j] < 0:
                result_R[i][j] = 0
            if result_R[i][j] > 255:
                result_R[i][j] = 255
    merged = cv2.merge([result_B, result_G, result_R])
    print(result_B)
    cv2.imshow("result", result_B)
    cv2.waitKey(0)


def main():
    img_lenna = cv2.imread("lenna.jpg", cv2.IMREAD_COLOR)  # 512 x 512
    img_lenna = cv2.resize(img_lenna, dsize=(128, 128), interpolation=cv2.INTER_AREA)
    img_landscape = cv2.imread("landscape.jpg", cv2.IMREAD_COLOR)
    img_landscape = cv2.resize(
        img_landscape, dsize=(128, 128), interpolation=cv2.INTER_AREA
    )  # 512 x 512
    img_wonjin = cv2.imread("wonjin.jpg", cv2.IMREAD_COLOR)
    img_wonjin = cv2.resize(
        img_wonjin, dsize=(128, 128), interpolation=cv2.INTER_AREA
    )  # 512 x 512
    dct(img_landscape)


if __name__ == "__main__":
    main()

