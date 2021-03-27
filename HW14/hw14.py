import cv2
import numpy as np
from sklearn.cluster import MeanShift, KMeans
import csv

# land1 cluster : 16   land2 cluster : 20

# import deepcopy
def cluster():
    bgr = cv2.imread("land2.jpg")
    bgr = cv2.resize(bgr, dsize=(100, 100), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(lab)
    cv2.imwrite("land2_original.jpg", bgr)
    X = []
    for i in range(100):
        for j in range(100):
            X.append([L[i][j], A[i][j], B[i][j]])
    # print("ok")
    clustering = MeanShift(bandwidth=10).fit(X)  # label 18개
    # data_labels = clustering.labels_
    # data_labels = data_labels.reshape(100, 100)
    # f = open("labels2.csv", "w", encoding="utf-8", newline="")
    # wr = csv.writer(f)
    # for i in range(100):
    #     wr.writerow(data_labels[i])

    # mean_per_label = [[0, 0, 0, 0] for _ in range(20)]
    # labels = []
    # f = open("labels2.csv", "r", encoding="utf-8")
    # rdr = csv.reader(f)
    # for line in rdr:
    #     labels.append(line)

    # for i in range(100):
    #     for j in range(100):
    #         label = labels[i][j]
    #         label = int(label)
    #         mean_per_label[label][0] += L[i][j]
    #         mean_per_label[label][1] += A[i][j]
    #         mean_per_label[label][2] += B[i][j]
    #         mean_per_label[label][3] += 1

    # for lab in mean_per_label:
    #     lab[0] /= lab[3]
    #     lab[1] /= lab[3]
    #     lab[2] /= lab[3]

    # for i in range(100):
    #     for j in range(100):
    #         label = int(labels[i][j])
    #         L[i][j] = mean_per_label[label][0]
    #         A[i][j] = mean_per_label[label][1]
    #         B[i][j] = mean_per_label[label][2]

    for i in range(100):
        for j in range(100):
            L[i][j] = clustering[i * 100 + j][0]
            A[i][j] = clustering[i * 100 + j][1]
            B[i][j] = clustering[i * 100 + j][2]
    merged_lab = cv2.merge((L, A, B))
    clustered_BGR = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite("clustered_land2_temp.jpg", clustered_BGR)
    # K-Means

    # kmeans_model = KMeans(n_clusters=20)
    # kmeans_model.fit(X)
    # kmeans_labels = kmeans_model.labels_
    # kmeans_labels = kmeans_labels.reshape(100, 100)
    # f = open("Kmeans2.csv", "w", encoding="utf-8", newline="")
    # wr = csv.writer(f)
    # for i in range(100):
    #     wr.writerow(kmeans_labels[i])
    # f.close()

    # mean_per_label = [[0, 0, 0, 0] for _ in range(20)]
    # labels = []
    # f = open("Kmeans2.csv", "r", encoding="utf-8")
    # rdr = csv.reader(f)
    # for line in rdr:
    #     labels.append(line)

    # for i in range(100):
    #     for j in range(100):
    #         label = labels[i][j]
    #         label = int(label)
    #         mean_per_label[label][0] += L[i][j]
    #         mean_per_label[label][1] += A[i][j]
    #         mean_per_label[label][2] += B[i][j]
    #         mean_per_label[label][3] += 1

    # for lab in mean_per_label:
    #     lab[0] /= lab[3]
    #     lab[1] /= lab[3]
    #     lab[2] /= lab[3]

    # for i in range(100):
    #     for j in range(100):
    #         label = int(labels[i][j])
    #         L[i][j] = mean_per_label[label][0]
    #         A[i][j] = mean_per_label[label][1]
    #         B[i][j] = mean_per_label[label][2]

    # merged_lab = cv2.merge((L, A, B))
    # kmeans_BGR = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    # cv2.imwrite("kmeans_land2.jpg", kmeans_BGR)
    # f.close()


cluster()
