import numpy as np
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
from scipy import io
import numpy as np
from spectral import *
import spectral as sp
import matplotlib.pyplot as plt
import cv2
import time
import glob
import os
import pandas as pd
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import PCA, NMF
from datetime import datetime


from model import DDCNN
from dataset import set_datasets, Dataset, calibration



if __name__=="__main__":

    std_list = [0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    
    # std_list = [0.00001]
    abs_path = "C:/Users/EL-DESKTOP-7/Desktop/test/experiment_set"
    data_paths = ["dataset4"]
    # data_paths = ["dataset3"]
    # data_paths = [ "dataset1", "dataset2", "dataset3"]
    for data_path in data_paths:
        for std in std_list:
            experiment_num = 3
            for i in range(experiment_num):  
                # train_data
                train_folder = f"{abs_path}/{data_path}/train"
                train_calibrations = []
                train_labels = []
                
                for file in os.listdir(train_folder):
                    # print(f"file = {file}") 
                    directory = os.path.join(train_folder, file)
                    # print(f"directory = {directory}")
                    if 'LL5XI.JPG' in file:
                        continue
                    train_label = np.load(f'{directory}/label.npy') 
                    train_data = np.array(sp.io.envi.open(directory + '/data.hdr', directory + '/data.raw').load())
                    train_dr = np.array(sp.io.envi.open(directory + '/DARKREF.hdr', directory + '/DARKREF.raw').load()) 
                    train_wr = np.array(sp.io.envi.open(directory + '/WHITEREF.hdr', directory + '/WHITEREF.raw').load()) 
                    train_calibration = calibration(train_data, train_dr, train_wr)
                    train_calibrations.append(train_calibration)
                    train_labels.append(train_label)
                train_calibrations=np.array(train_calibrations)
                train_labels=np.array(train_labels)
                
            
                # test_data
                test_folder = f"{abs_path}/{data_path}/test"
                test_calibrations = []
                test_labels = []
                
                for file in os.listdir(test_folder):
                    # print(f"file = {file}")  
                    directory = os.path.join(test_folder, file)
                    if 'LL5XI.JPG' in file:
                        continue
                    test_label = np.load(f'{directory}/label.npy') 
                    test_data = np.array(sp.io.envi.open(directory + '/data.hdr', directory + '/data.raw').load())
                    test_dr = np.array(sp.io.envi.open(directory + '/DARKREF.hdr', directory + '/DARKREF.raw').load()) 
                    test_wr = np.array(sp.io.envi.open(directory + '/WHITEREF.hdr', directory + '/WHITEREF.raw').load()) 
                    test_calibration = calibration(test_data, test_dr, test_wr)
                    test_calibrations.append(test_calibration)
                    test_labels.append(test_label)
                test_calibrations=np.array(test_calibrations)
                test_labels=np.array(test_labels)
                

                # validation_data
                validation_folder = f"{abs_path}/{data_path}/validation"
                validation_calibrations = []
                validation_labels = []
                
                for file in os.listdir(validation_folder):
                    directory = os.path.join(validation_folder, file)
                    if 'LL5XI.JPG' in file:
                        continue
                    validation_label = np.load(f'{directory}/label.npy') 
                    validation_data = np.array(sp.io.envi.open(directory + '/data.hdr', directory + '/data.raw').load())
                    validation_dr = np.array(sp.io.envi.open(directory + '/DARKREF.hdr', directory + '/DARKREF.raw').load()) 
                    validation_wr = np.array(sp.io.envi.open(directory + '/WHITEREF.hdr', directory + '/WHITEREF.raw').load()) 
                    validation_calibration = calibration(validation_data, validation_dr, test_wr)
                    validation_calibrations.append(validation_calibration)
                    validation_labels.append(validation_label)
                validation_calibrations=np.array(validation_calibrations)
                validation_labels=np.array(validation_labels)
            

                # ddcnn_parameter
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(device)
                batch_size = 100
                learning_rate = 0.001
                patch_size = 5
                ic = 224
                oc = 16
                k = 3 
                s = 1
                oc2 = 128
                oc3 = 32
                k2 = 1 
                g = 32
                t_ic = oc+6*g
                t_oc = int(t_ic/2)
                c_oc = t_oc  + 16*g
                epoch_num = 100
                num_layer1 = 6
                
                
                
                ddcnn = DDCNN(ic, oc, oc2, k, s, oc3, k2, g, t_ic, t_oc, c_oc, num_layer1)
                ddcnn = ddcnn.to(device)
                optimizer = optim.Adam(ddcnn.parameters(), lr = learning_rate)
                criterion = nn.CrossEntropyLoss()
                
                # train_dataloader
                train_indices = set_datasets(train_calibrations, train_labels, patch_size=patch_size)
                train_dataset = Dataset(train_calibrations, train_labels, train_indices, std=std, patch_size=patch_size)
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)    

                # test_dataloader
                test_indices = set_datasets(test_calibrations, test_labels, patch_size=patch_size)
                test_dataset = Dataset(test_calibrations, test_labels, test_indices, std=0, patch_size=patch_size)
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                # valid_dataloader
                valid_indices = set_datasets(validation_calibrations, validation_labels, patch_size=patch_size)
                valid_dataset = Dataset(validation_calibrations, validation_labels, valid_indices, std=0, patch_size=patch_size)
                valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
                
                
                # train
                start_now = datetime.now()
                start_now = start_now.today().strftime("%y-%m-%d-%H:%M:%S")  
                print(f"{start_now} {data_path} std={std} Train start ")
                start = time.time()
                running_loss = 0.0
                best_accuracy = 0.0
                best_valid = np.inf
                n = len(train_dataloader)
                for epoch in range(epoch_num):
                    
                    running_accuracy = 0.0
                    total = 0
                    print(f"{epoch+1}/{epoch_num}")
                    ddcnn.train()
                    for _, (train_data, train_label) in enumerate(train_dataloader):
                        train_data = train_data.to(device)
                        train_label = train_label.to(device)
                        optimizer.zero_grad() #gradient 초기화
                        output = ddcnn(train_data) 
                        loss = criterion(output, train_label)
                        loss.backward() # gradient 계산
                        optimizer.step() # gradient 업데이트
                        running_loss += loss.item()
                        train_loss = running_loss/n
                        # print(f"running_loss={running_loss} / train_loss={train_loss}")
                            
                    
                    # validation        
                    with torch.no_grad():
                        ddcnn.eval()
                        val_loss=0
                        for _, (valid_data, valid_label) in enumerate(valid_dataloader):
                            valid_data = valid_data.to(device)
                            valid_label = valid_label.to(device)
                            output = ddcnn(valid_data)
                            val_loss += criterion(output, valid_label)
                            _, predicted = torch.max(output, 1)
                            total += valid_label.size(0)
                            running_accuracy += (predicted == valid_label).sum().item()
                        
                    accuracy = (100*running_accuracy / total)
                    
                    # if accuracy > best_accuracy:
                    #     path = f"C:/intern/ddcnn/models/Model.pth"
                    #     torch.save(ddcnn.state_dict(), path)
                    #     best_accuracy = accuracy
                    #     print(f"best_epoch is {epoch+1}")
                        
                    if val_loss < best_valid:
                        path = f"C:/intern/ddcnn/models/Model_{data_path}.pth"
                        torch.save(ddcnn.state_dict(), path)
                        best_valid = val_loss
                        print(f"best_epoch is {epoch+1}")
                    
                    now = datetime.now()
                    now = now.today().strftime("%y-%m-%d-%H:%M:%S")    
                    print(f"[epoch{(epoch+1)}] {now} accuracy : {accuracy:.3f} / val_loss : {val_loss:.3f}")      
                        
                # test
                correct = 0
                predictions = []  
                test_labels = []    
                
                print("Test start")
                path = f"C:/intern/ddcnn/models/Model_{data_path}.pth"
                ddcnn.load_state_dict(torch.load(path))
                
                with torch.no_grad():
                    ddcnn.eval()
                    for _, (test_data, test_label) in enumerate(test_dataloader):
                        test_data = test_data.to(device)
                        test_label = test_label.to(device)
                        test_label = test_label.to(torch.float)
                        # print(f"test_label={test_label}")
                        # print(f"test_label={np.shape(test_label)}")
                        output = ddcnn(test_data)  
                        prediction = output.data.max(1)[1]
                        # _, prediction = torch.max(output, 1)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        _, prediction2 = torch.max(probabilities, 1)
                        predictions.extend(prediction.cpu().numpy())
                        test_labels.extend(test_label.cpu().numpy())

                    # Confusion matrix 
                    matrix = confusion_matrix(predictions, test_labels) # for문밖에
                    precision_score1 = precision_score(test_labels, predictions, average='weighted')
                    recall_score1 = recall_score(test_labels, predictions, average='weighted')
                    f1_score1 = f1_score(test_labels, predictions, average='weighted')
                    # auc = roc_auc_score(test_labels, predictions, multi_class='ovr')
                    print(f"Confusion Matrix: \n {matrix}")
                    print(f"precision:{precision_score1:.3f}")
                    print(f"recall:{recall_score1:.3f}")
                    print(f"f1score:{f1_score1:.3f}")
        
                    
                    #OA, AA, kappa print 
                    OA = np.sum(np.diag(matrix))/ np.sum(matrix)
                    print(f'OA: {OA:.2f}')
                    AA = np.mean(np.diag(matrix)/np.sum(matrix, axis=0))
                    print(f'AA: {AA:.2f}')
                    
                    total_sum = np.sum(matrix)
                    # matrix_sum = (np.sum(matrix, axis=1) * np.sum(matrix, axis=0)) / total_sum ** 2 
                    # kappa = (OA - np.sum(matrix_sum)) / (1 - np.sum(matrix_sum)+1e-12)
                    matrix_sum = (np.matmul(np.sum(matrix, axis=0),np.sum(matrix, axis=1))) / total_sum ** 2 
                    kappa = (OA - matrix_sum) / (1 - matrix_sum+1e-12)
                    
                    time2 = time.time()- start
                    print(f'kappa: {kappa:.2f}')
                    print(f"time : {time2:.2f}")
                    
                        
                    # Result save
                    file_name = f"C:/intern/ddcnn/results.csv"
                    now = datetime.now()
                    now = now.today().strftime("%y-%m-%d-%H:%M:%S")
                    
                    if not os.path.exists(file_name):
                            result_data = {'Date':[], 'Dataset':[], 'Test': [], 'Std':[], 'Epochs': [], 'OA': [], 'AA': [], 'Kappa': [], 'Time': [], 'Precision':[], 'Recall':[], 'F1score':[], 'patch_size':[] }
                                
                    else:
                        existing_results = pd.read_csv(file_name)
                        result_data = {
                            'Date' : list(existing_results['Date']),
                            'Dataset' : list(existing_results['Dataset']),
                            'Test' : list(existing_results['Test']),
                            'Std': list(existing_results['Std']),
                            'Epochs': list(existing_results['Epochs']),
                            'OA': list(existing_results['OA']),
                            'AA': list(existing_results['AA']),
                            'Kappa': list(existing_results['Kappa']),
                            'Time': list(existing_results['Time']),
                            'Precision': list(existing_results['Precision']),
                            'Recall': list(existing_results['Recall']),
                            'F1score': list(existing_results['F1score']),
                            'patch_size' : list(existing_results['patch_size'])
                        }
                    result_data['Date'].append(now)
                    result_data['Dataset'].append(f"{data_path}")
                    result_data['Test'].append(i+1)
                    result_data['Std'].append(std)
                    result_data['Epochs'].append(epoch_num)
                    result_data['OA'].append(f"{OA:.3f}")
                    result_data['AA'].append(f"{AA:.3f}")
                    result_data['Kappa'].append(f"{kappa:.3f}")
                    result_data['Time'].append(f"{time2:.3f}")
                    result_data['Precision'].append(f"{precision_score1:.3f}")
                    result_data['Recall'].append(f"{recall_score1:.3f}")
                    result_data['F1score'].append(f"{f1_score1:.3f}")
                    result_data['patch_size'].append(patch_size)
                    result_df = pd.DataFrame(result_data)
                    result_df.to_csv(file_name, index=False)
                    
                    #image         
                    # dataiter = iter(test_dataloader)
                    # images, labels = next(dataiter)
                    # random_indices = np.random.choice(len(predictions), 1, replace=False)
                    # for i, idx in enumerate(random_indices):
                    #     image_to_display = images[i][0].cpu().numpy()  
                    #     plt.figure(figsize=(5, 5))
                    #     plt.imshow(image_to_display, cmap='gray')  
                    #     plt.show()

                # main2_function(i, epoch_num)