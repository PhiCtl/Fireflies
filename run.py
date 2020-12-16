from Load import load_test_data
from Train import predict

#load data
print("Folder of pose tracking and annotation files (absolute path from data_fly folder): ")
folder = input()
X,Y = load_test_data(folder)

print(
'''
Enter flag for model prediction: \n 
##################################
-> LSTM         : to make predictions with best LSTM model
-> TCN          : to make predictions with best TCN model
-> Random_Forest: to make predictions with best Random Forest model
##################################
'''
)
flag = input()

#make predictions
predict(X,Y, flag)


