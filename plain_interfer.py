import pandas as pd
import torch
from hosptialapp.verifiable_cnn1 import *

def compare(loglist):
    start = int(loglist[-1])
    end = int(loglist[0])

    path_plain = '/home/data/train.csv'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn=CNN()
    cnn.load_state_dict(torch.load('/home/www/project/hospital_2/hosptialapp/model_parameter.pkl3'))

    data_p = pd.read_csv(path_plain, index_col=0, skiprows=lambda x: x < start or x > end+1)
    all_data_p = []
    for i in data_p.values:
        signal = i[0].split(',')
        all_data_p += [signal]
    all_data_p = pd.DataFrame(all_data_p)
    all_data_p = all_data_p.astype('float')  # float64
    train_x = torch.FloatTensor(all_data_p.values)
    train_x = train_x.to(device)
    train_x = train_x.reshape([-1, 1, 205])  # 128*1*205
    train_x = cnn.conv_1(train_x)  # 128*16*102
    train_x = cnn.avepool_1(train_x)  # 128*16*51
    train_x = cnn.conv_2(train_x)  # 128*16*25
    train_x = cnn.avepool_2(train_x)  # 128*16*13

    train_x = train_x.view(train_x.size(0), -1)
    # x = self.flatten(x)
    x_p = cnn.fc_2(train_x)
    x_p = x_p.detach().numpy().tolist()
    return x_p