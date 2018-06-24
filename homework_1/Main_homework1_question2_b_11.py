# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:58:04 2017

@author: Emmanuele
"""
#%%
import importlib
import numpy as np
import pandas as pd
import Functions_homework1_11 as lb
importlib.reload(lb)
 #%%
# Create the dataset split in train & test
x_train, x_test, y_train, y_test = lb.generateTrainingTestSets()
#%%
# =============================================================================
# all_loss=pd.read_excel("Main_homework1_question1_b_11.xlsx")            
# optim=all_loss[all_loss['loss_test'] == min(all_loss['loss_test'])].round(6)
# print(optim)
# =============================================================================
#%%
# grid paramenter
X=pd.concat((x_train,x_test))
loss_final=pd.DataFrame(np.zeros((1,6)))
loss_final.columns=("N_Neurons","rho","sigma","loss","loss_test","time")
#%%
RBF=lb.NeuralNetwork(N_samples=x_train.shape[0],
            						N_NeuronsPerLayer=30,
                            max_epochs=10000,
            						rho=0.0001,
            						sigma=0.5,
                            display_step=1000,
                            learning_rate=0.1)

RBF.RBF_config_unsupervised(centers=lb.getCenters(X,RBF.N_NeuronsPerLayer))
# fill the matrix of the information
loss_final.iloc[0,:]=RBF.run(x_train, x_test, y_train, y_test)

#%%
output = open("output_homework1_11.txt","a")
output.write("This is Homework 1, Question 2, Point 2")
output.write("\nTraining objective function," + "%f" % loss_final.loss.values[0])
output.write("\nTest MSE," + "%f" % loss_final.loss_test.values[0])
output.write("\nTraining computing time," + "%f" % loss_final.time.values[0])
output.write("\nFunction evaluations," + "%i" % RBF.iter)
output.write("\nGradient evaluations," + "%i" % RBF.iter)
output.close()
