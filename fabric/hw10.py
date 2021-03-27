import cv2
import numpy as np
from PIL import Image
import random


# GRAYSCALE 이미지 만들기
# image_set = []
# for i in range(1, 21):
#     image_set.append(str(i) + ".jpg")

# for image in image_set:
#     img = cv2.imread("./"+image, cv2.IMREAD_GRAYSCALE)
#     # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite("./gray_" + image, img)

# center = int(length/2)
# img = Image.open("gray_20.jpg")
# img = img.resize((128, 128))
# area = (32, 32, 96, 96)
# cropped_img = img.crop(area)
# img.save("20_size.jpg")
# img.show()
# cropped_img.show()

# 이미지 완성


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    for i in range(len(flat)):
        flat[i] = abs(flat[i])
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


largest_indices_with_patterns = []
for i in range(1, 21):
    img = cv2.imread("./"+str(i)+"_size.jpg", 0)
    m_sum = np.zeros((64, 64))
    for j in range(10):
        r_row = random.randint(0, 64)
        r_col = random.randint(0, 64)
        img_crop = img[r_row:r_row+64, r_col:r_col+64]
        # print(img_crop)
        f = np.fft.fft2(img_crop)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        m_sum += magnitude
    m_average = m_sum/10
    row_arr, col_arr = largest_indices(m_average, 11)
    temp_list = []
    for k in range(11):
        temp_list.append((row_arr[k], col_arr[k]))
    largest_indices_with_patterns.append(temp_list)
# Average Magnitude 완성

statistics = [[0, 0, 0] for _ in range(20)]

for i in range(20):
    statistics[i][0] = i

# statistic의 #1 은 사진 번호, #2 는 count sum, #3은 random하게 나온횟수
for k in range(1000):
    pattern_num = random.randint(1, 20)
    # print("pattern : ", pattern_num)
    pattern_img = cv2.imread("./"+str(pattern_num)+"_size.jpg", 0)
    random_row = random.randint(0, 64)
    random_col = random.randint(0, 64)
    pattern_block = pattern_img[random_row:random_row +
                                64, random_col:random_col+64]
    f = np.fft.fft2(pattern_block)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    arr1, arr2 = largest_indices(magnitude, 10)
    largest_indices_result = []
    for i in range(10):
        largest_indices_result.append((arr1[i], arr2[i]))

    result = (-1, -1)
    for i in range(20):
        indices = largest_indices_with_patterns[i]
        count = 0
        for (row, col) in largest_indices_result:
            if (row, col) in indices:
                count += 1
        (idx, cnt) = result
        if count > cnt:
            result = (i, count)
    if pattern_num != result[0]+1:
        print("Recognition Error !!")
    else:
        statistics[result[0]][1] += result[1]-1
        statistics[result[0]][2] += 1
        # print("계산결과 pattern : ", result[0]+1, result[1]-1)
print(statistics)
for i in range(20):
    print(str(i+1) + "번째 사진은 1000번중 랜덤하게 " +
          str(statistics[i][2]) + "번 나왔으며 가장 큰 coefficients 10개 중 평균 " + str(statistics[i][1]/statistics[i][2]) + "개가 똑같다.")
