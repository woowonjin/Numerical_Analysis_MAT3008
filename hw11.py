import cv2


def correlation(img):
    (img_B, img_G, img_R) = cv2.split(img)


for i in range(1, 11):
    img = cv2.imread(str(i) + ".jpg", cv2.IMREAD_COLOR)
    img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_AREA)
    cv2.imshow("1", img)
    cv2.waitKey(0)
