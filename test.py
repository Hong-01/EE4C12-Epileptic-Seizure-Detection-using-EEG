import os
print("hello world")
#你好，世界
data_dir = os.path.join(os.getcwd(), 'E:\DATA\TUD\Master\TUD_Master_Y1\Q1\EE4C12 Machine Learning For Electrical Engineering\CodeLab\Lab4\dataset\\')
print('Number of images in the dataset folder: ', len(os.listdir(data_dir)))