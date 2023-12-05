import numpy as np, cv2
import random
import copy

def k_means(image, k, max_iteration, threshold):

    random_centroid_list = []
    
    # cluster 배열
    cluster_array = [[] for _ in range(k)]
    
    #랜덤하게 초기 중심점을 구함
    for i in range(k):
         random_row = random.randint(0,image.shape[0] -1)
         random_col = random.randint(0,image.shape[1]- 1)
         random_centroid_list.append(image[random_row][random_col])
         
    centroid_array = np.array(random_centroid_list)

    previous_centroid_array = copy.deepcopy(centroid_array)
        
    for iter in range(max_iteration):
            #각 중심점에 할당    
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    cluster_comparison_array = np.array([])
                    
                    for l in range(k):
                        cluster_comparison_array = np.append(cluster_comparison_array, manhattan(image[i][j], centroid_array[l]))
                        
                min_index = np.argmin(cluster_comparison_array)
                cluster_array[min_index].append(image[i][j])
            
            # 클러스터의 평균 값으로 중심점 업데이트
            for c in range(k):
                    if cluster_array[c]:
                        centroid_array[c] = np.sum(cluster_array[c], axis = 0) / len(cluster_array[c])

            # 중심점 이동 확인
            movement = np.sum(np.abs(previous_centroid_array - centroid_array))

            # 어느 정도 이동했으면 수렴으로 간주
            if movement < threshold:
                break
            
            #이전 중심값 저장
            previous_centroid_array = copy.deepcopy(centroid_array)
    
    cluster_array = [np.array(cluster, dtype='uint8') for cluster in cluster_array]

    return centroid_array, cluster_array
        
        
def manhattan(image,randomCentroidVetor):
    
    manhattan_distance = np.sum(np.abs(image - randomCentroidVetor), axis=None)
    
    return manhattan_distance        
    


def main():
    image = cv2.imread("images/BSDS1.png")

    # 군집화된 이미지를 저장할 배열
    clustering_image = np.zeros_like(image)

    k = 2

    #최대 반복수
    max_iteration = 1000
    
    # 최적의 클러스터 대표값, 클러스터링 배열
    optimal_centroid, cluster_array = k_means(image, k, max_iteration, threshold=0.001)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            for d in range(len(cluster_array)):
                for f in cluster_array[d][:]:
                    if np.array_equal(image[i][j], f):
                        clustering_image[i][j] = optimal_centroid[d]


    cv2.imshow("result_image",clustering_image)
    cv2.waitKey(0)
                            
if __name__ == "__main__":
    main()
    