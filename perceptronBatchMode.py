import pandas as pd
import numpy as np

def read_csv_file(file_path):
    
    csv = pd.read_csv(file_path)
    
    labels = []
    
    for label in csv['activity']:
        labels.append(label)
        
    datas = []
    
    for data in range(len(csv)):
        datas.append([csv['acceleration_x'][data],
                     csv['acceleration_y'][data],
                     csv['acceleration_z'][data],
                     csv['gyro_x'][data],
                     csv['gyro_y'][data],
                     csv['gyro_z'][data]])
        
    return labels, datas

def predict(y):
    
    if y <= 0:
        return 0
    else:
        return 1
        
def main():
    
    w = np.random.rand(1,6)
    b = -0.01
    p = 0.001
    
    batch_size = 1000
    
    correct = 0
    wrong = 0
    
    file_name = 'data/행동분류_데이터.csv'
    labels, datas = read_csv_file(file_name)
    
    datas = np.array(datas)
    
    for i in range(0, len(datas), batch_size):
        batch_datas = datas[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        y = np.sum(batch_datas * w, axis=1) + b
        predictions = np.array([predict(y_val) for y_val in y])

        correct += np.sum(predictions == batch_labels)
        wrong += np.sum(predictions != batch_labels)

        w += p * np.sum((batch_labels - predictions)[:, None] * batch_datas, axis=0)
        b += p * np.sum(batch_labels - predictions)
                
    print(f'정인식률: {round(((correct/(correct+wrong))*100),3)}, 가중치:{w}, 편향: {round(b,3)}')
             
if __name__ == "__main__":
    main()