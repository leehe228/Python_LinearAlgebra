""" Linear Algebra Project 1
    202011353 이호은 """

import cv2
import numpy as np


# N by N Haar Matrix 생성
def MakeNHaarMatrix(n):
    array_A = np.array([[1], [1]])
    array_B = np.array([[1], [-1]])

    # n이 1이면
    if n == 1:
        return np.array([1])

    # n이 2 이상이면
    else:
        m = 1
        H_m = np.array([1])
        for i in range(2, n + 1):
            new_array = np.hstack([np.kron(H_m, array_A), np.kron(np.eye(m), array_B)])
            m *= 2
            if m == n:
                return new_array
            else:
                H_m = new_array


# N x N 행렬에서 좌측 상단 k x k만큼 잘라 행렬에 넣고 다시 복원
def MakeCroppedImage(B, normalized_H_n, k, n):
    # 파일 이름 설정
    file_name = "result_" + str(k) + ".bmp"

    # Crop (자르기)
    Bhat = np.zeros((n, n))
    temp_Bhat = B[0: int(n / k), 0: int(n / k)]

    for i in range(int(n / k)):
        for j in range(int(n / k)):
            Bhat[i][j] = temp_Bhat[i][j]

    # 복원
    Ahat = (normalized_H_n.dot(Bhat)).dot(normalized_H_n.T)

    # 복원된 A hat 출력
    print("======================================")
    print(f"Ahat Matrix of result_{k} : \n{Ahat}")

    # 이미지 파일로 저장
    cv2.imwrite(file_name, Ahat)
    print(f"\n{file_name} created successfully!")

    print("======================================", end='\n\n')
    # 종료


# 이미지 출력 함수
def ShowImage(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)


# 결과 이미지 출력 함수
def showResult():
    for i in range(1, 9):
        file_name = "result_" + str(2 ** i) + ".bmp"
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

        cv2.imshow(file_name, img)
        cv2.waitKey(0)
        cv2.destroyWindow(file_name)


if __name__ == '__main__':
    # 이미지 읽기
    original_image = cv2.imread('image_lena_24bit.bmp', cv2.IMREAD_GRAYSCALE)

    height, width = original_image.shape
    print(f"height : {height}, width : {width}", end='\n')
    cv2.imshow("original", original_image)
    cv2.waitKey(0)
    cv2.destroyWindow("original")

    # 사진 크기 추출
    n = height

    # Numpy Array 로 변환
    original_array = np.array(original_image)
    print(f"Original Photo Array : \n{original_array}")

    # N by N 크기의 Haar Matrix 생성
    H_n = MakeNHaarMatrix(n)
    print(f"N Haar Matrix (H_n) : \n{H_n}")

    # Normalization
    normalized_H_n = np.zeros((n, n))
    for i in range(n):
        s = 0
        for j in range(n):
            if H_n[j][i] != 0:
                s += 1
        for j in range(n):
            normalized_H_n[j][i] = H_n[j][i] / (s ** 0.5)

    # DHWT : B = H^TAH
    B = (normalized_H_n.T.dot(original_array)).dot(normalized_H_n)
    print(f"B (DHWT) : \n{B}")

    # 1/2 ~ 1/256 크롭 이미지 복원 및 저장
    for i in range(1, 9):
        MakeCroppedImage(B, normalized_H_n, 2 ** i, n)

    # 사진 순서대로 출력
    showResult()
