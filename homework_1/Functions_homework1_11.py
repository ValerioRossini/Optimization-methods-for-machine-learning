import numpy as np
import time
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from matplotlib import cm
import random
np.set_printoptions(precision=2)
random.seed(1613638)

def grad(learning_rate):
	#initialite Gradient Optimazer
	#return tf.train.AdagradOptimizer(learning_rate)
	#return tf.train.AdadeltaOptimizer(learning_rate,rho=0.95, epsilon=1e-06)
	#return tf.train.RMSPropOptimizer(learning_rate)
	return tf.train.MomentumOptimizer(learning_rate,momentum=0.8)
	
def converge(grad,tol=1e-3):
	return np.sum(grad**2)<tol

def franke_function(x1, x2):
  return (
    .75 * np.exp(-(9 * x1 - 2) ** 2 / 4.0 - (9 * x2 - 2) ** 2 / 4.0) +
    .75 * np.exp(-(9 * x1 + 1) ** 2 / 49.0 - (9 * x2 + 1) / 10.0) +
    .5 * np.exp(-(9 * x1 - 7) ** 2 / 4.0 - (9 * x2 - 3) ** 2 / 4.0) -
    .2 * np.exp(-(9 * x1 - 4) ** 2 - (9 * x2 - 7) ** 2))

def tanh(t,sigma):
    return (1-tf.exp(-sigma*t))/(1+tf.exp(-sigma*t))

def generateTrainingTestSets(Test_size=0.3,n=100,plot3d_printoption=False):
	x_dat=pd.DataFrame(np.random.rand(n,2))
	y_dat=pd.DataFrame(franke_function(x_dat[0],x_dat[1])+np.random.uniform(-0.01,+0.01,x_dat[0].shape[0]))
	x_train, x_test, y_train, y_test=train_test_split(x_dat, y_dat, test_size=Test_size)
	if plot3d_printoption:
		plot3D(x_train,y_train)
	return x_train, x_test, y_train, y_test
	
def scatterY(y,y_pred,y_test,y_test_pred):
	plt.figure(1)
	plt.subplot(121)
	plt.xlim(0,  1.2)
	plt.ylim(0, 1.2)
	plt.xlabel("y Train")
	plt.ylabel("y predicted")
	plt.title("y Train  vs y pred")
	plt.scatter(y,y_pred)
	plt.subplot(122)
	plt.xlim(0,  1.2)
	plt.ylim(0, 1.2)
	plt.xlabel("y Test")
	plt.ylabel("y predicted")
	plt.title("y Test vs y pred")
	plt.scatter(y,y_pred)
	plt.show()		
	
def plot3D(x,y):
	x1_scatter=x[0]
	x2_scatter=x[1]
	y_scatter=y[0]
	
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	x1 = x2 = np.arange(0, 1, 0.01)
	xx1, xx2 = np.meshgrid(x1, x2)
	yy =franke_function(xx1,xx2)

	surf = ax.plot_surface(xx1, xx2, yy, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

	ax.scatter(x1_scatter,x2_scatter,y_scatter, c='darkgreen',linewidth=0)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.title("3D plot using the Franke\'s function")
	plt.show()
	
def gaussian_function(x,sigma,centers)	:
	samples = x
	centroids = centers
	expanded_vectors = tf.expand_dims(samples, 0)
	expanded_centroids = tf.expand_dims(centroids, 1)
	distances = tf.reduce_sum( tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
	initial = tf.transpose(tf.exp(-tf.pow(distances, 2)/sigma))
	return(initial)

def getCenters(x,n=10):
	estimator = KMeans(n_clusters=n)
	estimator.fit(x)
	return np.array(estimator.fit(x).cluster_centers_,dtype='float32')
    
def predict(x,W_in,b_in,W_out,sigma):
	# Training computation.
	logits_1 = tf.matmul(x,W_in)-b_in
	tanh_layer=tanh(logits_1,sigma)
	logits_2 = tf.matmul(tanh_layer,W_out)
	# Predictions for the training
	y = logits_2
	return y
	
def predict_RBF(x,W_out,sigma,centers):
	# Training computation.
	RBF_layer=gaussian_function(x,sigma,centers)
	logits_2 = tf.matmul(RBF_layer,W_out)
	# Predictions for the training
	y = logits_2
	return y
		
def loss_func(y,y_true,rho,P,W_in=0.,b_in=0.,W_out=0.,c=0.):
	loss =  (1/2)*tf.reduce_mean(tf.squared_difference(y, y_true))
	# Loss function with L2 Regularization with rho
	regularizers = tf.reduce_sum(W_in**2)+tf.reduce_sum(b_in**2) + tf.reduce_sum(W_out**2) + tf.reduce_sum(c**2)
	loss = tf.reduce_sum(loss + rho * regularizers)
	return loss
	
def createPair(MLP,x_train):
	d_in=MLP.N_Features 
	with MLP.g.as_default():

		#Wessels and Barnard
		r1,r2=-(3/np.sqrt(d_in)),(3/np.sqrt(d_in))
		W_in_WB=tf.Variable(tf.random_uniform([MLP.N_Features, MLP.N_NeuronsPerLayer],r1,r2))
		b_in_WB=tf.Variable(tf.random_uniform([MLP.N_NeuronsPerLayer],r1,r2))

		# Le Cun
		r3,r4=-np.sqrt(d_in),np.sqrt(d_in) #this range favors that the activation function works in its active region.
		W_in_LC=tf.Variable(tf.random_uniform([MLP.N_Features, MLP.N_NeuronsPerLayer],r3,r4))
		b_in_LC=tf.Variable(tf.random_uniform([MLP.N_NeuronsPerLayer],r3,r4))

		#Yam and Chow
		b=np.sqrt(np.max(x_train[0])-min(x_train[0])+np.max(x_train[1])-min(x_train[1]))
		r5,r6=-(8.72/b)*np.sqrt(3/2),(8.72/b)*np.sqrt(3/d_in)
		W_in_YC=tf.Variable(tf.random_uniform([MLP.N_Features, MLP.N_NeuronsPerLayer],r5,r6))
		b_in_YC=tf.Variable(tf.random_uniform([MLP.N_NeuronsPerLayer],r5,r6))

		#Haffner
		W_in_H=tf.Variable(tf.random_normal([MLP.N_Features, MLP.N_NeuronsPerLayer]))
		b_in_H=tf.Variable(tf.random_uniform([MLP.N_NeuronsPerLayer]))

		#Nguyen & Widrow
		w_initial=tf.Variable(tf.random_uniform([MLP.N_Features, MLP.N_NeuronsPerLayer],-1,1))
		beta=tf.pow(0.7,(1/d_in))
		gamma=tf.sqrt(tf.reduce_sum(w_initial**2))
		W_in_NW=(beta*w_initial)/gamma

		b_initial=tf.Variable(tf.random_uniform([MLP.N_NeuronsPerLayer],-1,1))
		gamma=tf.sqrt(tf.reduce_sum(b_initial**2))
		b_in_NW=(beta*b_initial)/gamma

	return [[W_in_WB,b_in_WB],[W_in_LC,b_in_LC],[W_in_YC,b_in_YC],[W_in_H,b_in_H],[W_in_NW,b_in_NW]]
	
def gridsearch_par_MLP(x_train, x_test, y_train, y_test,eta=0.1,display=200):
	# grid paramenter
	n_ne_vec=np.array([2,5,10,15,20,30,50,70])
	rho_vec=np.array([0.0001,0.0005,0.00001],dtype='float32')
	sigma_vec=np.array([0.5,1,2.5,5],dtype='float32')

	all_loss=pd.DataFrame(np.zeros(((len(n_ne_vec)*len(rho_vec)*len(sigma_vec)),6)))
	all_loss.columns=("N_Neurons","rho","sigma","loss","loss_test","time")

	idx=0

	for j in np.arange(len(n_ne_vec)):
		for i in np.arange(len(rho_vec)):
			for q in np.arange(len(sigma_vec)):
				print("number",idx,"of", all_loss.shape[0])
				# Parameters for structure
				MLP=NeuralNetwork(N_samples=x_train.shape[0],
										N_NeuronsPerLayer=n_ne_vec[j],
										max_epochs=10000,
										rho=rho_vec[i],
										sigma=sigma_vec[q],
										display_step=display,
										learning_rate=eta)
				
				MLP.MLP_config()
				# fill the matrix of the information
				all_loss.iloc[idx,:]=MLP.run(x_train, x_test, y_train, y_test)
				
				idx+=1
            
	#%%
	writer= pd.ExcelWriter("Main_homework1_question1_a_11.xlsx")            
	all_loss.to_excel(writer,"Sheet1",)
	writer.save()

def gridsearch_par_RBF(x_train, x_test, y_train, y_test,eta=0.1,display=200):
		
	# grid paramenter
	n_ne_vec=np.array([2,5,10,15,20,30,50,70])
	rho_vec=np.array([0.0001,0.0005,0.00001],dtype='float32')
	sigma_vec=np.array([0.5,1,2.5,5],dtype='float32')

	all_loss=pd.DataFrame(np.zeros(((len(n_ne_vec)*len(rho_vec)*len(sigma_vec)),6)))
	all_loss.columns=("N_Neurons","rho","sigma","loss","loss_test","time")
	
	idx=0

	for j in np.arange(len(n_ne_vec)):
		for i in np.arange(len(rho_vec)):
			for q in np.arange(len(sigma_vec)):
				print("number",idx,"of", all_loss.shape[0])
				# Parameters for structure
				RBF=NeuralNetwork(N_samples=x_train.shape[0],
										N_NeuronsPerLayer=n_ne_vec[j],
										max_epochs=10000,
										rho=rho_vec[i],
										sigma=sigma_vec[q],
										display_step=display,
										learning_rate=eta)
				
				RBF.RBF_config_supervised()
				# fill the matrix of the information
				all_loss.iloc[idx,:]=RBF.run(x_train, x_test, y_train, y_test)
				idx+=1
	#%%
	writer= pd.ExcelWriter("Main_homework1_question1_b_11.xlsx")            
	all_loss.to_excel(writer,"Sheet1",)
	writer.save()

def gridsearch_range_MLP(x_train, x_test, y_train, y_test,eta=0.1,display=200):
	all_loss1=pd.read_excel("Main_homework1_question1_a_11.xlsx")            
	optim1=all_loss1[all_loss1['loss_test'] == min(all_loss1['loss_test'])].round(6)
	#%%
	MLP=NeuralNetwork(N_samples=x_train.shape[0],
										N_NeuronsPerLayer=optim1["N_Neurons"].values[0],
										max_epochs=10000,
										rho=optim1["rho"].values[0],
										sigma=optim1["sigma"].values[0],
										display_step=display,
										learning_rate=eta)
	#%%
	rangePair=createPair(MLP,x_train)
	#%%
	all_loss=pd.DataFrame( np.zeros((len(rangePair),6)))
	all_loss.columns=("N_Neurons","rho","sigma","loss","loss_test","time")
	idx=0
	#%%
	for pair in rangePair:
		print("number",idx,"of", all_loss.shape[0])
		MLP.MLP_twoblock_config(W_in=pair[0],b_in=pair[1])
		# fill the matrix of the information
		all_loss.iloc[idx,:]=MLP.run(x_train, x_test, y_train, y_test)
		idx+=1
	all_loss["Name of the Pair"]=["Wessels and Barnard","Le Cun","Yam and Chow","Haffner","Nguyen & Widrow"]     
	#%%
	writer= pd.ExcelWriter("Main_homework1_question2_a_11.xlsx")            
	all_loss.to_excel(writer,"Sheet1",)
	writer.save() 
	
	
class NeuralNetwork:
	
	def __init__(self,N_samples,
						N_NeuronsPerLayer,
						display_step=1,
						max_epochs = 10000,
						learning_rate = 0.01,
						rho=0.001,
						sigma=1):
		self.g=tf.Graph()
		with self.g.as_default():
			# Parameters for structure
			self.N_samples=N_samples
			self.N_Features = 2
			self.N_HiddenLayers = 1
			self.N_NeuronsPerLayer = N_NeuronsPerLayer #vector
			self.N_output = 1
			# All the train test, in this way iteration=epoch
			self.max_epochs = max_epochs
			# learning rate
			self.learning_rate = learning_rate
			# weight of regularization term
			self.rho=rho
			# Parameter for acivation function
			self.sigma=sigma
			# How many epoch for print
			self.display_step = display_step
		
		# define input and output matrices like placeholder
			self.x = tf.placeholder(tf.float32, [None,self.N_Features])
			self.x_te = tf.placeholder(tf.float32, [None,self.N_Features])
			self.y_true = tf.placeholder(tf.float32, [None,self.N_output])
			self.y_te_true=tf.placeholder(tf.float32, [None,self.N_output])	
			
			self.twodecomp=False
		
	def MLP_config(self):
		
		with self.g.as_default():
			W_in=tf.Variable(tf.truncated_normal([self.N_Features, self.N_NeuronsPerLayer]))
			b_in=tf.Variable(tf.zeros([1,self.N_NeuronsPerLayer]))
			W_out=tf.Variable(tf.truncated_normal([self.N_NeuronsPerLayer,self.N_output]))
				
			self.y_pred=predict(self.x,
										W_in,
										b_in,
										W_out,
										self.sigma)
										
			self.loss=loss_func(y=self.y_pred,
										y_true=self.y_true,
										rho=self.rho,
										P=self.N_samples,
										W_in=W_in,
										b_in=b_in,
										W_out=W_out)
			
			# # Computing the gradient of cost with respect to W and b
			# grad_W, grad_b, grad_V = tf.gradients(xs=[W_in,b_in,W_out], ys=self.loss)
			
			# # Gradient Step
			# new_W = W_in.assign(W_in - self.learning_rate * grad_W)
			# new_b = b_in.assign(b_in - self.learning_rate * grad_b)
			# new_V=W_out.assign(W_out -self.learning_rate* grad_V)
			
					
			
			Optimizer=grad(self.learning_rate)
			
			w, b, v = Optimizer.compute_gradients(var_list=[W_in,b_in,W_out], loss=self.loss)
			self.omega=tf.concat([w[0],b[0],tf.transpose(v[0])],axis=0)
			self.opt=Optimizer.apply_gradients([w,b,v])		
			
			##############################################################
			# Test computation.
			# Predictions for the training
			self.y_pred_test = predict(self.x_te,
												W_in,
												b_in,
												W_out,
												self.sigma)
												
			self.loss_test=loss_func(y=self.y_pred_test,
										y_true=self.y_te_true,
										rho=0,
										P=1)
			##############################################################
			
			self.init_op = tf.global_variables_initializer()
	
	def MLP_twoblock_config(self,W_in,b_in):
		with self.g.as_default():
		
			#W_in=tf.Variable(tf.random_uniform([self.N_Features, self.N_NeuronsPerLayer],a,b))
			#b_in=tf.Variable(tf.random_uniform([self.N_NeuronsPerLayer],a,b))
			W_out=tf.Variable(tf.zeros([self.N_NeuronsPerLayer,self.N_output]))
		
					
			self.y_pred=predict(self.x,
										W_in,
										b_in,
										W_out,
										self.sigma)
										
			self.loss=loss_func(y=self.y_pred,
										y_true=self.y_true,
										rho=self.rho,
										P=self.N_samples,
										W_out=W_out)
										
			Optimizer=grad(self.learning_rate)
			
			v = Optimizer.compute_gradients(var_list=[W_out], loss=self.loss)
			self.omega=v[0][0]
			self.opt=Optimizer.apply_gradients(v)		
			
			
			# Test computation.
			# Predictions for the training
			self.y_pred_test = predict(self.x_te,
												W_in,
												b_in,
												W_out,
												self.sigma)
												
			self.loss_test=loss_func(y=self.y_pred_test,
										y_true=self.y_te_true,
										rho=0,
										P=1)
			##############################################################
			# Compute the gradient 
			# self.opt = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss,var_list=[W_out])
			self.init_op = tf.global_variables_initializer()
			
	def MLP_decomposition_config(self):
	
		with self.g.as_default():
			self.twodecomp=True
			W_in=tf.Variable(tf.random_uniform([self.N_Features, self.N_NeuronsPerLayer]))
			b_in=tf.Variable(tf.zeros([1,self.N_NeuronsPerLayer]))
			W_out=tf.Variable(tf.ones([self.N_NeuronsPerLayer,self.N_output]))
		
			# prediction of y step 1		
			self.y_pred_1=predict(self.x,
										W_in,
										b_in,
										W_out,
										self.sigma)
									
			self.loss_v=loss_func(y=self.y_pred_1,
										y_true=self.y_true,
										rho=self.rho,
										P=self.N_samples,
										W_out=W_out)
										
			Optimizer=grad(self.learning_rate)
			
			v= Optimizer.compute_gradients(var_list=[W_out], loss=self.loss_v)
			
			self.opt_v=Optimizer.apply_gradients(v)			
			#opt_v = tf.train.GradientDescentOptimizer(self.learning_rate)
			#dW_out = opt_v.compute_gradients(self.loss_v,var_list=[W_out])[0][1]
			
			#self.opt_v = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss_v,var_list=[W_out])		
			# prediction of y step 2
			self.y_pred=predict(self.x,
										W_in,
										b_in,
										W_out,
										self.sigma)
									
			self.loss=loss_func(y=self.y_pred,
										y_true=self.y_true,
										rho=self.rho,
										P=self.N_samples,
										W_in=W_in,
										b_in=b_in)
										
			w, b= Optimizer.compute_gradients(var_list=[W_in,b_in], loss=self.loss)
			self.opt=Optimizer.apply_gradients([w,b])		
			
			self.omega=tf.concat([w[0],b[0],tf.transpose(v[0][0])],axis=0)
			#self.opt= tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss,var_list=[W_in,b_in])
			# Test computation###################################
			# Predictions for the training
			self.y_pred_test = predict(self.x_te,
												W_in,
												b_in,
												W_out,
												self.sigma)
												
			self.loss_test=loss_func(y=self.y_pred_test,
										y_true=self.y_te_true,
										rho=0,
										P=1)
			##############################################################
			# Compute the gradient 
			
			self.init_op = tf.global_variables_initializer()
	
	def RBF_config_supervised(self):
		with self.g.as_default():
			
			W_out=tf.Variable(tf.truncated_normal([self.N_NeuronsPerLayer,self.N_output]))
			centers=tf.Variable(tf.random_uniform([self.N_NeuronsPerLayer,self.N_Features]))
					
			self.y_pred=predict_RBF(self.x,
										W_out,
										self.sigma,
										centers)
																		
			self.loss=loss_func(y=self.y_pred,
										y_true=self.y_true,
										rho=self.rho,
										P=self.N_samples,
										W_out=W_out,
										c=centers)
			
			# Computing the gradient of cost with respect to W and b
			# grad_c, grad_V = tf.gradients(xs=[centers,W_out], ys=self.loss)
			
			# # Gradient Step
			# new_c = centers.assign(centers - self.learning_rate * grad_c)
			# new_V=W_out.assign(W_out -self.learning_rate* grad_V)
			
			# self.omega=tf.concat([grad_c,grad_V],axis=1)
			# self.opt=tf.concat([new_c,new_V],axis=1)
			
			Optimizer=grad(self.learning_rate)
			
			c, v = Optimizer.compute_gradients(var_list=[centers,W_out], loss=self.loss)
			self.omega=tf.concat([c[0],v[0]],axis=1)
			self.opt=Optimizer.apply_gradients([c,v])	
			# Test computation.
			# Predictions for the training
			self.y_pred_test = predict_RBF(self.x_te,
												W_out,
												self.sigma,
												centers)
												
			self.loss_test=loss_func(y=self.y_pred_test,
										y_true=self.y_te_true,
										rho=0,
										P=1)
										
			##############################################################
			# Compute the gradient 
			
			self.init_op = tf.global_variables_initializer()
			
	def RBF_config_unsupervised(self,centers):
		with self.g.as_default():
			
			W_out=tf.Variable(tf.truncated_normal([self.N_NeuronsPerLayer,self.N_output]))
		
					
			self.y_pred=predict_RBF(self.x,
										W_out,
										self.sigma,
										centers)
																			
			self.loss=loss_func(y=self.y_pred,
										y_true=self.y_true,
										rho=self.rho,
										P=self.N_samples,
										W_out=W_out)
										
			Optimizer=grad(self.learning_rate)
			
			v = Optimizer.compute_gradients(var_list=[W_out], loss=self.loss)
			self.omega=v[0][0]
			self.opt=Optimizer.apply_gradients(v)							
			# Test computation.
			# Predictions for the training
			self.y_pred_test = predict_RBF(self.x_te,
												W_out,
												self.sigma,
												centers)
												
			self.loss_test=loss_func(y=self.y_pred_test,
										y_true=self.y_te_true,
										rho=0,
										P=1)
			##############################################################
			# Compute the gradient 
			#self.opt = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss,var_list=[W_out])
			self.init_op = tf.global_variables_initializer()
			
	def run(self,x_train, x_test, y_train, y_test,plot3d_printoption=False,scatter_printoption=False):
		
		print("Start----------------------------------------------------------")
		print("This is the NN with ",self.N_NeuronsPerLayer," neurons")
		print("The weight of regularization term is ",self.rho)
		print("The sigma of the activation function is ",self.sigma)
		
		# Now we need to actually run the computational graph
		with tf.Session(graph=self.g) as sess:
			sess.run(self.init_op)
			time4 = time.time()
			self.iter=0.
			l_old=1000
			max_iter_without_progress=1000
			# we now run only the last operation in the chain above defined: tensorflow will compute everything as planned
			# we only need to feed the placeholders with the input-output examples
			while(True and  self.iter<self.max_epochs):
				# initialization of the inputs and outputs for process
				d={   self.x : x_train, 
						self.y_true : y_train,
						self.x_te : x_test, 
						self.y_te_true :y_test }
				
				if(self.twodecomp==True)	:
					_=sess.run(self.opt_v,feed_dict=d)
					
				_,l,y,l_test,y_t,omega = sess.run((self.opt,
																	self.loss,
																	self.y_pred,
																	self.loss_test,
																	self.y_pred_test,
																	self.omega),feed_dict=d)
				# Display logs per epoch step
				self.iter+=1
				if (self.iter) % self.display_step == 0:
					print("Epoch:", '%04d' % (self.iter+1), "cost=", '{:.9f}'.format(l_test))
				
				# se non ci sono miglioramenti nelle ultime 1000 iterazioni dell 5% chiudi
				if (self.iter) % max_iter_without_progress == 0:
					l_new=l
					diff=(l_new-l_old)/l_old
					if abs(diff)*100<5:
						print("Not improvement in the last ",max_iter_without_progress,"so I break the optimization")
						break
					l_old=l_new
				# se da nan = troppo difficile da calcolare il gradiente break	
				if(l_test!=l_test):
					print("Not able to compute gradient!!")
					break
				
				if (converge(omega)):
					print("Function evaluation",self.iter)
					print("Optimization Solved!")
					break
			                     
			time_gradient = time.time()-time4
			print("Time to compute the optimization: ", time_gradient)
			print("loss: ",l,"loss in the test",l_test)  
			print("end----------------------------------------------------------")
			
			y_p_final,y_p_t_final=pd.DataFrame(y),pd.DataFrame(y_t)
			
			if scatter_printoption:
				scatterY(y_train,y_p_final,y_test,y_p_t_final) 
			
			if plot3d_printoption:
				plot3D(x_train,y_p_final)
				plot3D(x_test,y_p_t_final)
			
			return np.array([self.N_NeuronsPerLayer,self.rho,self.sigma,l,l_test,time_gradient])	