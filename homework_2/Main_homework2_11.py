# Load the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import time 
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import importlib
import matplotlib.pyplot as plt
import Function_homework2_11 as lb
importlib.reload(lb)
import random
#%%
random.seed(1613638)

#min_max_scaler = preprocessing.MinMaxScaler()

# Read the dataset using pandas
df = pd.read_csv("2017.12.11 Dataset Project 2.csv")

# Separate the feautures from the labels
y = df["y"].map({0:-1,1:1}).values
x = df.drop("y", axis=1).values

x_scaled = scale(x)

# Split the dataset in x_train, x_test, y_train and y_test
x_train, x_test, y_train, y_test=train_test_split(x_scaled, y, test_size=0.30)

# Length of training and test set
p = x_train.shape[0]

#%%
# =============================================================================
# # GRID SEARCH WITH K-FOLD
# best_error_test = 100000
# C_list = np.array([0.5,0.8,1,2])
# gamma_list =  np.array([0.0003,0.0007,0.003,0.007,0.03,0.3,0.5, 0.8,1])
# startTime_total = time.time()
# allErrorsTest=[]
# allErrorsTrain=[]
# 
# for C in C_list:
#     for par in gamma_list:
#         print ("---------------------(C =",C,"and free parameter =",par,")---------------------")
#         
#         gamma_fix = par
#         c_fix = C
#         kf = KFold(n_splits=5, shuffle=True)
#         i=0
#         # Creation of the vector empty
#         vec_error=[]
#         vec_error_train=[]
#         vec_acc=[]
#         vec_timeElapsed=[]
#         
#         for train_index, validation_index in kf.split(x_train):
#             
#             print("---------------------",i+1,"FOLD---------------")
#             
#             X_train, X_valid = x_train[train_index], x_train[validation_index]
#             Y_train, Y_valid = y_train[train_index], y_train[validation_index]
#             
#             K_kf=lb.gaussian_kernel(X_train,X_train,gamma_fix)
#             # Q with gaussian       
#             Q_kf = lb.create_Q(Y_train,K_kf)
#             
#             startTime = time.time()
#             res_QP_kf=lb.QP_minimization(Q_fix=Q_kf,y=Y_train,C=c_fix,tol=1e-6)
#             lambda_star_kf = res_QP_kf.x
#             endTime = time.time()
#                    
#             print("------TRAIN-----------")
#             Y_pred=lb.pred(X_train,X_train,Y_train,lambda_star_kf,gamma_fix)
#             error_train=1-lb.compute_accuracy(Y_train,Y_pred)
#             print("Error",error_train)
#             print("Accurancy",lb.compute_accuracy(Y_train,Y_pred))
#             
#             print("------TEST-----------")
#             Y_pred_valid=lb.pred(X_valid,X_train,Y_train,lambda_star_kf,gamma_fix)
#             error_test=1-lb.compute_accuracy(Y_valid,Y_pred_valid)
#             
#             # fill the vector
#             vec_error_train.append(error_train)
#             vec_error.append(error_test)
#             vec_acc.append(lb.compute_accuracy(Y_valid,Y_pred_valid))
#             vec_timeElapsed.append(endTime - startTime)
#            
#             print("Error",vec_error[i])
#             print("Accurancy",vec_acc[i])
#             i+=1
#             
#         print ("--------------------------------------------------------------")
#         allErrorsTest.append(np.mean(vec_error))
#         allErrorsTrain.append(np.mean(vec_error_train))
#         print("The overall train error is ",np.mean(vec_error_train))   
#         print("The overall test accurancy is ",np.mean(vec_acc))   
#         print("The overall test error is ",np.mean(vec_error))
#         print("The overall time elapsed is ",np.mean(vec_timeElapsed))
#         print()
# 
#         if best_error_test > np.mean(vec_error):
#             best_error_test = np.mean(vec_error)
#             list_variables_test = [C, par, np.mean(vec_error_train),best_error_test, np.mean(vec_timeElapsed),np.mean(vec_acc), res_QP_kf]
# #%%
# print ("--------------------------------------------------------------")
# print()
# print("Best C:", list_variables_test[0])
# print("Best gamma:", list_variables_test[1])
# print("Error on train:", list_variables_test[2])
# print("Error on test:", list_variables_test[3])
# print("Accurancy on test:", list_variables_test[5])
# print("Elapsed time:", list_variables_test[4])
# print()
# endTime_total = time.time()
# timeElapsed_total = endTime_total - startTime_total
# print("Elapsed time total:", timeElapsed_total)
# #%%
# error_test_matrix =np.array(allErrorsTest).reshape(len(C_list),len(gamma_list))
# error_train_matrix =np.array(allErrorsTrain).reshape(len(C_list),len(gamma_list))
# dfIterTest = pd.DataFrame(error_test_matrix)
# dfIterTrain = pd.DataFrame(error_train_matrix)
# dfIterTest.columns = ["p: "+str(l) for l in gamma_list]  
# dfIterTest.index = ["C: "+str(l) for l in C_list] 
# dfIterTrain.columns = ["p: "+str(l) for l in gamma_list]  
# dfIterTrain.index = ["C: "+str(l) for l in C_list]
# 
# i=0
# plt.figure(figsize=(12,6))
# for num in range(len(C_list)):
#     fig=plt.subplot(221+i)
#     plt.plot(gamma_list, error_test_matrix[num,:],marker="o",label="test error")
#     plt.plot(gamma_list, error_train_matrix[num,:],marker="o",label="train error")
#     plt.title(("Errors with C = %.2f" %C_list[num]),fontsize = 15)
#     plt.xlabel('gamma',fontsize = 10)
#     plt.ylabel('errors',fontsize = 10)
#     plt.legend(fontsize = 8,loc="best")
#     i+=1
# plt.tight_layout()
# =============================================================================
#%%
# 1 point
# Compute the Kernel and Q matrix
gamma_opt = 0.1 # list_variables_test[1]
C_opt= 1#list_variables_test[0]
K_RBF=lb.gaussian_kernel(x_train,x_train,gamma_opt)
# Q with gaussian       
Q_RBF = lb.create_Q(y_train,K_RBF)
print(Q_RBF)
#%%
print ("---------------------(C =",C_opt,"and free parameter =",gamma_opt,")---------------------")
print()
startTime = time.time()
res_QP_gauss=lb.QP_minimization(Q_fix = Q_RBF , y = y_train, C = C_opt,tol=1e-6)
endTime = time.time()
timeElapsed = endTime - startTime
lambda_star_RBF = res_QP_gauss.x
#%%
print("------TEST-----------")
y_pred_test=lb.pred(x_test,x_train,y_train,lambda_star_RBF,gamma_opt)
print("Error on test",1-lb.compute_accuracy(y_test,y_pred_test))
print("Accurancy on test",lb.compute_accuracy(y_test,y_pred_test))

print("------TRAIN-----------")
y_pred=lb.pred(x_train,x_train,y_train,lambda_star_RBF,gamma_opt)
print("Error on train",1-lb.compute_accuracy(y_train,y_pred))
print("Accurancy on train",lb.compute_accuracy(y_train,y_pred))
print("--------------------------------")
print ("Elapsed time:", timeElapsed)
print()
print("Lambda (first 10) founded with Gaussian Kernel")
print(lambda_star_RBF[:10])
print()
#%%
output = open("output_homework2_11.txt","a")
output.write("Homework 2, Question 1")
output.write("\nTraining objective function," + "%f" % res_QP_gauss.fun)
output.write("\nTest accuracy," + "%f" % lb.compute_accuracy(y_test,y_pred_test))
output.write("\nTraining computing time," + "%f" % timeElapsed)
output.write("\nFunction evaluations," + "%i" % res_QP_gauss.nfev)
output.write("\nGradient evaluations," + "%i" % res_QP_gauss.njev)
output.close()
#%%
random.seed(1613638)
# Compute the decomposition method
C_DM = C_opt
eps_DM=0.1
lambda_DM,res=lb.decomposition_method(Q_RBF,y_train,C_DM,eps_DM,q=6,max_iter=300) 
#%%
print()
print("Lambda (first 10) founded with analytic solution of the SVM light")
print(lambda_DM[:10])
print()
print("------TRAIN-----------")
y_pred=lb.pred(x_train,x_train,y_train,lambda_DM,gamma_opt)
print("Error on train",1-lb.compute_accuracy(y_train,y_pred))
print("Accurancy on train",lb.compute_accuracy(y_train,y_pred))

print("------TEST-----------")
y_pred_test=lb.pred(x_test,x_train,y_train,lambda_DM,gamma_opt)

print("Error on test",1-lb.compute_accuracy(y_test,y_pred_test))
print("Accurancy on test",lb.compute_accuracy(y_test,y_pred_test))
#%%
output = open("output_homework2_11.txt","a")
output.write("\nHomework 2, Question 2")
output.write("\nTraining objective function," + "%f" % res[0].fun)
output.write("\nTest accuracy," + "%f" % lb.compute_accuracy(y_test,y_pred_test))
output.write("\nTraining computing time," + "%f" % res[3])
output.write("\nOuter iterations," + "%i" % res[4])
output.write("\nFunction evaluations," + "%i" % res[1])
output.write("\nGradient evaluations," + "%i" % res[2])
output.close()
#%%
random.seed(1613638)
# MVP method
C_MVP = C_opt
lambda_MVP, time, outerIterations=lb.MVP_method(Q_RBF,y_train,C_MVP,max_iter=500)

#%%
print()
print("Lambda (first 10) founded with analytic solution of the MVP")
print(lambda_MVP[:10])
print()

print("------TRAIN-----------")
y_pred=lb.pred(x_train,x_train,y_train,lambda_MVP,gamma_opt)
print("Error on train",1-lb.compute_accuracy(y_train,y_pred))
print("Accuracy on train",lb.compute_accuracy(y_train,y_pred))

print("------TEST-----------")
y_pred_test=lb.pred(x_test,x_train,y_train,lambda_MVP,gamma_opt)
print("Error on test",1-lb.compute_accuracy(y_test,y_pred_test))
print("Accuracy on test",lb.compute_accuracy(y_test,y_pred_test))
#%%
from functools import reduce
x=lambda_MVP
c = np.repeat(1,p)
elem=[x.T,Q_RBF,x]
obj = 0.5 * reduce(np.dot, elem) - np.dot(c, x)    

#%%
output = open("output_homework2_11.txt","a")
output.write("\nHomework 2, Question 3")
output.write("\nTraining objective function," + "%f" % obj)
output.write("\nTest accuracy," + "%f" % lb.compute_accuracy(y_test,y_pred_test))
output.write("\nTraining computing time," + "%f" % time)
output.write("\nOuter iterations," + "%i" % outerIterations)
output.close()