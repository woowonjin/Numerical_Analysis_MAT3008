from scipy import io
import numpy as np
from matplotlib import pyplot as plt


def main():
    # 파일의 데이터 가져오기
    mat_file = io.loadmat("YaleB.mat")
    data_set = mat_file["fea"]
    # 사진 입력 받이서 float형태 행렬도 만들기
    all_faces_temp = np.ndarray.flatten(data_set)[: 1024 * 1774]
    test_faces_temp = np.ndarray.flatten(data_set)[1024 * 1774 :]
    all_faces_temp = all_faces_temp.astype(np.float)
    test_faces_temp = test_faces_temp.astype(np.float)

    # 데이터는 1774개로 정했고, 원본 데이터의 형식을 맞추기 위해 일단 이렇게 설정, 또 테스트용 50개의 vector를 만들기
    all_faces = all_faces_temp.reshape(1774, 1024)
    test_faces = test_faces_temp.reshape(640, 1024)

    # 입력 받은 사진이 옆으로 누워있어서 바로 세우기
    for i in range(0, 1774):
        all_faces[i] = all_faces[i].reshape(32, 32).transpose().flatten()
    for i in range(0, 640):
        test_faces[i] = test_faces[i].reshape(32, 32).transpose().flatten()

    # mean 구하기
    mean_arr = []
    for i in range(0, 1024):
        sum = 0
        for j in range(0, 1774):
            sum += all_faces[j][i]
        mean_arr.append(sum / 1774)
    mean_face = np.array(mean_arr)
    for i in range(0, 1774):
        all_faces[i] -= mean_face

    all_faces = all_faces.transpose()

    # 중앙값 얼굴 나타내기
    # mean_face = mean_face.reshape(32, 32)
    # img = plt.imshow(mean_face)
    # img.set_cmap("gray")
    # plt.show()

    # A^t@A 로 square matrix 만들기
    # all_faces_square = all_faces @ all_faces.transpose()
    # eigen_values, eigen_vectors = np.linalg.eig(all_faces_square)

    # Matrix를 svd 분할
    u, s, vt = np.linalg.svd(all_faces)
    # img = plt.imshow(u[:, 0].reshape(32, 32))
    # img.set_cmap("gray")
    # plt.show()

    # eigen value가 큰 eigen vector을 위해 idx를 리스트로 나타내려고 하는데 자동으로 정렬되는거같다.
    # eigen_idx = []
    # for i in range(0, 40):
    #     max_eigen = 0
    #     idx = -1
    #     for j in range(0, 1024):
    #         if max_eigen < s[j]:
    #             if j in eigen_idx:
    #                 continue
    #             else:
    #                 idx = j
    #                 max_eigen = s[j]
    #     eigen_idx.append(idx)

    # eigenVector img 보기
    # u = u.transpose()
    # for i in range(0, 40):
    #     img = plt.imshow(u[i].reshape(32, 32))
    #     img.set_cmap("gray")
    #     plt.show()

    # eigen vector들을 모아놓은 matrix를 계산하기 쉽도록 transpose 시킨다.
    u = u.transpose()
    all_faces = all_faces.transpose()
    weights_all = []
    for k in range(0, 10):
        for i in range(0, 5):
            weights = []
            # test_face = test_faces[i] - mean_face
            test_face = test_faces[k * 64 + i] - mean_face
            for j in range(0, 40):
                weights.append(u[j].transpose() @ test_face)
            # print("coefficients of ", i, " : ", weights)
            weights_all.append(weights)
            face = mean_face
            for j in range(0, 40):
                face += weights[j] * u[j]
            # img_origin = plt.imshow((test_faces[k * 64 + i]).reshape(32, 32))
            # img_origin.set_cmap("gray")
            # plt.show()

            # img = plt.imshow(face.reshape(32, 32))
            # img.set_cmap("gray")
            # plt.show()
    weights_all = np.array(weights_all)
    weights_all = weights_all.astype(np.float)
    weights_mean = []
    weights_variance = []
    for i in range(0, 10):
        temp_weight = []
        mean_weight = (
            weights_all[i * 5]
            + weights_all[i * 5 + 1]
            + weights_all[i * 5 + 2]
            + weights_all[i * 5 + 3]
            + weights_all[i * 5 + 4]
        )
        mean_weight /= 5
        weights_mean.append(mean_weight)

    all_variances = []
    for i in range(0, 10):
        variance = []
        for j in range(0, 40):
            var = 0
            for k in range(0, 5):
                var += (weights_all[i * 5 + k][j] - weights_mean[i][j]) * (
                    weights_all[i * 5 + k][j] - weights_mean[i][j]
                )
            var /= 5
            variance.append(np.sqrt(var))
        all_variances.append(variance)
    all_variances = np.array(all_variances)
    print(all_variances)


if __name__ == "__main__":
    main()
