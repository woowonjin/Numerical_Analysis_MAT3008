import cv2  # openCV

# Importante!!! open CV usa modelo BGR e nao RGB
from dct import dct_2d
from idct import idct_2d


def main():
    # ler imagem.
    # A imagem lida eh um array[linha][coluna]
    img_lenna = cv2.imread("lenna.jpg", cv2.IMREAD_COLOR)  # 512 x 512
    # img_lenna = cv2.resize(img_lenna, dsize=(128, 128), interpolation=cv2.INTER_AREA)
    lenna_dct = dct_2d(img_lenna, 16)
    # cv2.imwrite("dct256.jpg", imgResult)
    lenna_idct = idct_2d(lenna_dct)
    cv2.imshow("lenna", idct_img)
    cv2.waitKey(0)
    # cv2.imwrite("idct256.jpg", idct_img)


if __name__ == "__main__":
    main()

