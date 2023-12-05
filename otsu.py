import numpy as np
import cv2
import matplotlib.pyplot as plt

# 누적 분포합
def calculate_cdf(hist):
    cdf = hist.cumsum()
    return cdf / cdf[-1]

def otsu(hist):
    norm_hist = hist / np.sum(hist)
    # 누적 분포합
    cdf = calculate_cdf(norm_hist)
    print(cdf)

    min_fn = float('inf')
    calculated_threshold = 0
    
    for t in range(1, len(hist)):
        w0 = np.sum(norm_hist[:t])
        w1 = np.sum(norm_hist[t:])
        q0, q1 = cdf[t], 1 - cdf[t]
        
        if q0 == 0:
             q0 = 0.00000001
        if q1 == 0:
             q1 = 0.00000001
        
        m0 = np.sum((np.arange(t) * norm_hist[:t]) / w0)
        m1 = np.sum((np.arange(t, len(norm_hist)) * norm_hist[t:]) / w1)
            
        v0 = np.sum((np.arange(t) - m0)**2 * w0) / q0
        v1 = np.sum((np.arange(t, len(norm_hist)) - m1)**2 * w1) / q1
        
        fn = q0 * v0 + q1 * v1
        
        if min_fn > fn:
            min_fn = fn
            calculated_threshold = t
            
    return calculated_threshold

#histogram 시각화
def visualize_histogram_and_threshold(hist, threshold):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(hist)), hist, color='gray')
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.plot(range(len(hist)), hist, color='gray', label='Histogram')
    plt.axvline(x=threshold, color='red', linestyle='--', label='Threshold')
    plt.title('Histogram with Otsu Threshold')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.show()
    
        
def main():
    image = cv2.imread("C:/hw/vision/images/ostu_test_image2.png", cv2.IMREAD_GRAYSCALE)
    copy_image = np.zeros_like(image)
    
    hist = np.zeros(256, dtype='uint8')
    
    #image 평탄화
    flatten_image = image.flatten().astype('uint8')
    
    for elem in flatten_image:
        hist[elem] += 1
     
    #픽셀 값이 count되어진 리스트 출력    
    print(hist)
    
    thres = otsu(hist)
    
    print(f"Threshold: {thres}")
    
    for i in range(copy_image.shape[0]):
        for j in range(copy_image.shape[1]):
            if image[i][j] > thres:
                copy_image[i][j] = 255
            else:
                copy_image[i][j] = 0
    
    visualize_histogram_and_threshold(hist, thres)
    
    cv2.imshow("Original", image)
    cv2.imshow("Changed", copy_image)
    cv2.waitKey(0) 

if __name__ == "__main__":
    main()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def histogram_stretching(image_path):
#     # 이미지 불러오기
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     # 히스토그램 스트레칭
#     stretched_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

#     return stretched_image

# def otsu(hist):
#     hist_norm = hist.ravel()/hist.max()
#     CDF = hist_norm.cumsum()
#     #initialization
#     bins = np.arange(256)
#     fn_min = np.inf
#     thresh = -1
#     #Otsu algorithm operation
#     for i in range(1,256):
#         p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
#         q1,q2 = CDF[i],CDF[255]-CDF[i] # cum sum of classes

#         if q1 == 0:
#             q1 = 0.00000001
#         if q2 == 0:
#             q2 = 0.00000001
#         b1,b2 = np.hsplit(bins,[i]) # weights
#         # finding means and variances
#         m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
#         v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
#         # calculates the minimization function
#         fn = v1*q1 + v2*q2
#         if fn < fn_min:
#             fn_min = fn
#             thresh = i

#     return thresh

# # histogram 시각화
# def visualize_histogram_and_threshold(hist, threshold):
#     plt.figure(figsize=(12, 5))

#     plt.subplot(1, 2, 1)
#     plt.bar(range(len(hist)), hist, color='gray')
#     plt.title('Histogram')
#     plt.xlabel('Pixel Value')
#     plt.ylabel('Frequency')

#     plt.subplot(1, 2, 2)
#     plt.plot(range(len(hist)), hist, color='gray', label='Histogram')
#     plt.axvline(x=threshold, color='red', linestyle='--', label='Threshold')
#     plt.title('Histogram with Otsu Threshold')
#     plt.xlabel('Pixel Value')
#     plt.ylabel('Frequency')
#     plt.legend()

#     plt.show()

# def main():
#     image_path = "C:/hw/vision/images/ostu_test_image2.png"

#     # 히스토그램 스트레칭
#     stretched_image = histogram_stretching(image_path)
#     copy_image = np.zeros_like(stretched_image)

#     # 오츄 알고리즘 적용
#     hist = cv2.calcHist([stretched_image], [0], None, [256], [0, 256]).flatten()
#     thres = otsu(hist)
    
#     for i in range(copy_image.shape[0]):
#          for j in range(copy_image.shape[1]):
#                 if stretched_image[i][j] > thres:
#                     copy_image[i][j] = 255
#                 else:
#                     copy_image[i][j] = 0
                    
#     print(thres)

#     # 시각화
#     visualize_histogram_and_threshold(hist, thres)

#     cv2.imshow("Original", cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
#     cv2.imshow("Stretched and Processed", copy_image)
#     cv2.waitKey(0)

# if __name__ == "__main__":
#     main()