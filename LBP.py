import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, class_names):
    # 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred)
    
    # 히트맵을 그리기 위한 설정
    plt.figure(figsize=(len(class_names), len(class_names)))
    
    # seaborn 라이브러리를 사용하여 히트맵 생성
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    
    # x축 레이블 설정
    plt.xlabel('Predicted')
    
    # y축 레이블 설정
    plt.ylabel('True')
    
    # 플롯 출력
    plt.show()


class LBP:
    def __init__(self, img):
        self.img = img

    def calculate_LBP(self, pixel_values):
        # LBP 연산 수행
        center = pixel_values[4]
        binary_values = (pixel_values > center).astype(np.uint8)
        binary_string = ''.join(np.array(binary_values).astype(str))
        decimal_value = int(binary_string, 2)
        return decimal_value

    def search_LBP(self):
        img = self.img
        lbp_img = np.zeros_like(img)        

        for i in range(img.shape[0]-2):
            for j in range(img.shape[1]-2):
                # 3x3 윈도우를 이용한 LBP 특성 추출
                pixel_values = img[i:i+3,j:j+3].flatten()
                lbp_value = self.calculate_LBP(pixel_values)
                lbp_img[i, j] = lbp_value

        return lbp_img

def lbp_histogram(image):
    lbp_obj = LBP(image)
    
    # LBP 이미지 생성 및 시각화 (추가된 부분)
    lbp_image = lbp_obj.search_LBP()

    # LBP 이미지 출력 부분
    # cv2.imshow("lbp_image", lbp_image)
    # cv2.waitKey(0)

    # 히스토그램 계산
    (hist, _) = np.histogram(lbp_image.ravel(), bins=np.arange(0, 256), range=(0, 256))
    # 히스토그램 정규화
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def load_images(folder_path, class_labels, binary_labels):
    # 데이터와 레이블을 저장할 리스트 초기화
    data = []
    labels = []

    # 클래스 라벨과 이진 라벨을 순회하면서 이미지 로딩
    for class_label, binary_label in zip(class_labels, binary_labels):
        # 클래스 폴더 경로 설정
        class_folder_path = os.path.join(folder_path, class_label)

        # 해당 클래스 폴더 내의 이미지 개수 계산
        image_count = len([f for f in os.listdir(class_folder_path) if f.endswith('.jpg')])

        # 각 이미지에 대한 처리
        for i in range(1, image_count + 1):
            # 이미지 파일 경로 설정
            img_path = f"{folder_path}/{class_label}/{i}.jpg"

            # 이미지 파일을 흑백으로 읽어옴
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # 이미지 로드 실패 시 건너뜀
            if img is None or img.size == 0:
                print(f"이미지 로드 실패: {img_path}")
                continue

            # LBP 히스토그램 계산
            lbp_hist = lbp_histogram(img)

            # 데이터와 레이블 리스트에 추가
            data.append(lbp_hist)
            labels.append(binary_label)

        # 클래스에 대한 이미지 로드 완료 메시지 출력
        print(f"'{class_label}' 이미지 로드 완료")

    # NumPy 배열로 변환하여 반환
    return np.array(data), np.array(labels)


def main():
    # 클래스 및 레이블 정의
    class_labels = ["bread", "wool", "sponge"]  # 빵, 울, 스펀지
    binary_labels = [0, 1, 2]  # 각 클래스에 대한 고유한 레이블

    # 훈련 세트 로드
    X_train, y_train = load_images("patterns/train", class_labels, binary_labels)


    # 검증 세트 로드
    X_valid, y_valid = load_images("patterns/valid", class_labels, binary_labels)


    # SVM 모델 정의 및 학습
    model = svm.SVC()
    model.fit(X_train, y_train)

    # 검증 세트로 모델 평가
    predictions = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, predictions)
    print("정확도:", accuracy)

    # 검증 세트에 대한 예측값과 실제 레이블을 사용하여 혼동 행렬 시각화
    plot_confusion_matrix(y_valid, predictions, class_labels)

if __name__ == "__main__":
    main()



