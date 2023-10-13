import csv
import numpy as np
from sklearn import model_selection, preprocessing
import pandas as pd
from core import GSAVES
from utils import degrade_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt

nb_sensors=32 # accepts values: 8, 16, or 32
if nb_sensors == 8:
    data = pd.read_csv('combined_floor8_8s.csv',header=None, skiprows=1)
elif nb_sensors == 16:
    data = pd.read_csv('combined_floor8_16s.csv', header=None, skiprows=1)
elif nb_sensors == 32:
    data = pd.read_csv('combined_floor8_32s.csv', header=None, skiprows=1)

sensors_data= pd.DataFrame(data)
print(len(sensors_data))
arr_sensors_data= sensors_data.iloc[:, :].values
arr_sensors_data=arr_sensors_data.astype(np.int)
# 20% of data for testing
val_frac=0.2
val_idx = int(len(arr_sensors_data) * (1 - val_frac))
x_train = arr_sensors_data[:val_idx]
x_test=arr_sensors_data[val_idx:]

# % of missing values
missingness= 0.5

true_positive=0
pred_positive=0
positive=0
train_batch_size=86400
offset=14400
test_len = len(x_test)
total_norm_t = []
a_losses_t = []
c_losses_t = []
r_losses_t = []
w_losses_t = []
epoch=0
first=True
for _ in range(epoch+1):
    train_len=len(x_train)
    i = 0
    begin=0
    end=train_batch_size
    while(train_len>=train_batch_size ):

        train_len-=(train_batch_size-offset)
        cx_tr, mask_tr = degrade_dataset(x_train[begin:end,:], missingness,  0)
        imputer = GSAVES(cx_tr,mask_tr,nb_sensors)
        if (first == True):
            total_norm, a_losses, c_losses, r_losses, w_losses=imputer.fit()
            first=False
        else:
            total_norm, a_losses, c_losses, r_losses, w_losses=imputer.fit(fine_tune=True)

        imputed_tr = imputer.transform().T
        imputed_tr = np.where(imputed_tr > 0.5, 1, 0)

        true_positive += (imputed_tr * x_train[begin:end, :]).sum()
        positive += x_train[begin:end, :].sum()
        pred_positive += imputed_tr.sum()

        total_norm_t.append(total_norm)
        a_losses_t.append(a_losses)
        c_losses_t.append(c_losses)
        r_losses_t.append(r_losses)
        w_losses_t.append(w_losses)
        begin = end-offset
        end = begin+train_batch_size
        i+=1

a_losses_t=np.array(a_losses_t)
c_losses_t=np.array(c_losses_t)
r_losses_t=np.array(r_losses_t)
w_losses_t=np.array(w_losses_t)
total_norm_t=np.array(total_norm_t)

a_losses_t=np.reshape(a_losses_t,-1)
c_losses_t=np.reshape(c_losses_t,-1)
r_losses_t=np.reshape(r_losses_t,-1)
w_losses_t=np.reshape(w_losses_t,-1)
total_norm_t=np.reshape(total_norm_t,-1)

fig, axs = plt.subplots(5, 1)
axs[0].plot(np.arange(0, len(a_losses_t), 1),a_losses_t)
axs[0].set_xlabel('iteration')
axs[0].set_ylabel('a_losses')
axs[0].grid(True)
axs[1].plot(np.arange(0, len(c_losses_t), 1),c_losses_t)
axs[1].set_xlabel('iteration')
axs[1].set_ylabel('c_losses')
axs[1].grid(True)
axs[2].plot(np.arange(0, len(r_losses_t), 1),r_losses_t)
axs[2].set_xlabel('iteration')
axs[2].set_ylabel('r_losses')
axs[2].grid(True)
axs[3].plot(np.arange(0, len(w_losses_t), 1),w_losses_t)
axs[3].set_xlabel('iteration')
axs[3].set_ylabel('w_losses')
axs[3].grid(True)
axs[4].plot(np.arange(0, len(total_norm_t), 1), total_norm_t)
axs[4].set_xlabel('iteration')
axs[4].set_ylabel('total_norm')
axs[4].grid(True)
fig.tight_layout()
plt.show()


precision= (true_positive / pred_positive)
recall = (true_positive / positive)
F = 2* ( (precision * recall) / ( precision + recall))
print('Recall:'+ str(recall))
print('Precision:'+ str(precision))
print('F score: '+ str(F))

i=0
true_positive=0
pred_positive=0
positive=0
active=0
stand_by=0
while( test_len>=train_batch_size):
    print(test_len)
    test_len -= train_batch_size
    begin = i * train_batch_size
    end = train_batch_size * (i + 1)

    cx_te, mask_te = degrade_dataset(x_test[begin:end, :], missingness, 0)
    imputed_te = imputer.transform_test(x_test[begin:end, :].T, mask_te.T).T
    imputed_te=np.where(imputed_te > 0.5, 1, 0)

    true_positive += (imputed_te *x_test[begin:end, :]).sum()
    positive += x_test[begin:end, :].sum()
    pred_positive += imputed_te.sum()

    active += cx_te.sum()
    stand_by += mask_te.sum() - cx_te.sum()
    i+=1

file=open("results_gsaves_1.txt", "w")
file.write("missingness"+str(missingness)+"\n")

precision= (true_positive / pred_positive)
recall = (true_positive / positive)
F = 2* ( (precision * recall) / ( precision + recall))
print('Recall:'+ str(recall))
print('Precision:'+ str(precision))
print('F score: '+ str(F))
file.write('Recall:'+ str(recall)+"\n"+'Precision:'+ str(precision)+"\n"+'F score: '+ str(F)+"\n")


total_readings= x_test.size

power = ((active * 0.092210723 + stand_by*0.00345 + (total_readings - active-stand_by) * 0.00216) * 3) /total_readings

lifetime=(240/(power/3)*0.85)/8760

print('total events:' + str(active) )
print('Total readings:'+str( total_readings))
print('Average energy consumption '+ str(power))
print('Liftime:'  + str( lifetime))

file.write('Average energy consumption '+ str(power)+"\n"+'Liftime:'  + str( lifetime)+"\n"+"\n")
file.close()

'''

imputed_tr =imputer.transform().T.round()
true_positive=0
pred_positive=0
positive=0

true_positive = (imputed_tr *x_train).sum()
positive = x_train.sum()
pred_positive = imputed_tr.sum()

recall = (true_positive / pred_positive)
precision = (true_positive / positive)
F = (5 * (precision * recall)) / (4 * precision + recall)
print(recall)
print(precision)
print(F)




imputed_tr = scaler_tr.inverse_transform(imputer.transform())


mae, rmse= imputation_accuracy(imputed_tr, x_train, np.linalg.inv(imputed_tr))
print(mae, rmse)


mae = mean_absolute_error(x_train,imputed_tr)
rmse = sqrt(mean_squared_error(x_train,imputed_tr))
print(mae)
print(sqrt)



imputer.add_data(oh_x_te,oh_mask_te,oh_num_mask_te,oh_cat_mask_te)

imputed_te = imputer.transform()
imputed_te = scaler_te.inverse_transform(imputed_te[x_train.shape[0]:])
imputer.fit(fine_tune=True)

imputed_te_ft = imputer.transform()
imputed_te_ft = scaler_te.inverse_transform(imputed_te_ft[x_train.shape[0]:])
'''