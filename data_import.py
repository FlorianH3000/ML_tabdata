"""
data import
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


label_encoder = LabelEncoder()


path = r'xxx.xlsx'

data_og = pd.read_excel(path)
data = data_og.to_numpy().transpose()


### data sample 1 ###
# data_1 = data[0]

### data sample 2 ###
# data_2 = data[1]

### data input ###

data1 = label_encoder.fit_transform(data[1])
data2 = label_encoder.fit_transform(data[2])
data3 = label_encoder.fit_transform(data[3])
data4 = label_encoder.fit_transform(data[4])
data5 = label_encoder.fit_transform(data[5])
data6 = label_encoder.fit_transform(data[6])
data7 = label_encoder.fit_transform(data[7])
data8 = label_encoder.fit_transform(data[8])
data9 = label_encoder.fit_transform(data[9])
data10 = label_encoder.fit_transform(data[10])
data11 = label_encoder.fit_transform(data[11])
data12 = label_encoder.fit_transform(data[12])
data13 = label_encoder.fit_transform(data[13])
data14 = label_encoder.fit_transform(data[14])
data15 = label_encoder.fit_transform(data[15])


data_input = np.stack(
    (
    data1,
    data2,
    data3,
    data4,
    data5,
    data6,
    data7,
    data8,
    data9,
    data10,
    data11,
    data12,
    data13,
    data14,
    data15,
    )).transpose()



# data_input = data[1:15].transpose()

### data label - adjust xxx ###
data_label = data[xxx]



