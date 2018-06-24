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

##lb.gridsearch_range_MLP(x_train, x_test, y_train, y_test,eta=0.1,display=200)

#%%
# =============================================================================
# all_loss=pd.read_excel("Main_homework1_question2_a_11.xlsx")            
# optim=all_loss[all_loss['loss_test'] == min(all_loss['loss_test'])].round(6)
# print(optim)
# =============================================================================
#%%
print("We choose Yam and Chow range") # dipende dal numero index
MLP=lb.NeuralNetwork(N_samples=x_train.shape[0],
            						N_NeuronsPerLayer=30,
                            max_epochs=10000,
            						rho=0.0001,
            						sigma=5,
                            display_step=500,
                            learning_rate=0.1)

Pair=lb.createPair(MLP,x_train)[0] ## 0=   Wessels and Barnard         
MLP.MLP_twoblock_config(W_in=Pair[0],b_in=Pair[1])
# fill the matrix of the information
loss_final=pd.DataFrame(np.zeros((1,6)))
loss_final.columns=("N_Neurons","rho","sigma","loss","loss_test","time")
loss_final.iloc[0,:]=MLP.run(x_train, x_test, y_train, y_test)

#%%
output = open("output_homework1_11.txt","a")
output.write("This is Homework 1, Question 2, Point 1")
output.write("\nTraining objective function," + "%f" % loss_final.loss.values[0])
output.write("\nTest MSE," + "%f" % loss_final.loss_test.values[0])
output.write("\nTraining computing time," + "%f" % loss_final.time.values[0])
output.write("\nFunction evaluations," + "%i" % MLP.iter)
output.write("\nGradient evaluations," + "%i" % MLP.iter)
output.close()

