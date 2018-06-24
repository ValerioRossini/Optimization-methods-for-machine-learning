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

# Create the dataset split in train & test
x_train, x_test, y_train, y_test = lb.generateTrainingTestSets()

##lb.gridsearch_par_MLP(x_train, x_test, y_train, y_test,eta=0.1,display=500)
#%%
# =============================================================================
# all_loss=pd.read_excel("Main_homework1_question1_a_11.xlsx")            
# optim=all_loss[all_loss['loss_test'] == min(all_loss['loss_test'])].round(6)
# print(optim)
# =============================================================================
#%%
MLP=lb.NeuralNetwork(N_samples=x_train.shape[0],
            						N_NeuronsPerLayer=30,
            						max_epochs=10000,
            						rho=0.0001,
            						sigma=1,
                            display_step=1000,
                            learning_rate=0.3)
            

MLP.MLP_config()
# fill the matrix of the information
loss_final=pd.DataFrame(np.zeros((1,6)))
loss_final.columns=("N_Neurons","rho","sigma","loss","loss_test","time")
loss_final.iloc[0,:]=MLP.run(x_train, x_test, y_train, y_test,False,False)

#%%
output = open("output_homework1_11.txt","a")
output.write("This is Homework 1, Question 1, Point 1")
output.write("\nTraining objective function," + "%f" % loss_final.loss.values[0])
output.write("\nTest MSE," + "%f" % loss_final.loss_test.values[0])
output.write("\nTraining computing time," + "%f" % loss_final.time.values[0])
output.write("\nFunction evaluations," + "%i" % MLP.iter)
output.write("\nGradient evaluations," + "%i" % MLP.iter)
output.close()
