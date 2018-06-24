#%%

import numpy as np
from scipy.optimize import minimize 
from functools import reduce
import time
from sklearn.metrics import accuracy_score

# Define as functions the kernel

def gaussian_kernel(xi,xj,g):
	p=xi.shape[0]
	p2=xj.shape[0]
	K2=np.zeros((p,p2))
	for i in range(p):
		for j in range(p2):
			K2[i,j]=np.exp(-g*(np.linalg.norm(xi[i]-xj[j])**2))
	return K2
    

def polynomial_kernel(xi,xj,t):
    return (np.inner(xi,xj)+1)**t

def create_Q(y,Kernel):
    # Q with polinomial
    p=Kernel.shape[0]
    Q = np.zeros((p,p))
    for i in range(0,p):
        for j in range(0,p):
            Q[i,j] = y[i] * y[j]* Kernel[i,j]
    return Q     

def function_to_minimize_QP(x,Q):
	c = np.repeat(1,Q.shape[0])
	elem=[x.T,Q,x]
	return 0.5 * reduce(np.dot, elem) - np.dot(c, x)

# Jacobian (gradient) of objective function. 
def jacobian_QP(x,Q):
	c = np.repeat(1,Q.shape[0])
	return np.dot(x.T, Q) - c

def QP_minimization(Q_fix,y,C=1,tol=1e-6):
    p=Q_fix.shape[0]
	
    cons = ({'type':'eq', 'fun':lambda t: np.dot(y,t)})
    
    x0 = np.repeat(0,p)

    bnds = tuple((0,C) for i in range(p))

    res_SLSQP = minimize(fun = function_to_minimize_QP, 
							x0 = x0, 
							args= (Q_fix),
							jac = jacobian_QP ,
							constraints = cons, 
							bounds = bnds, 
							method ='SLSQP', options={'ftol': tol, 'disp': True})
    print(res_SLSQP.message)
    
    return res_SLSQP

def compute_w(idx,lambda_star,y_train,Kernel):
	s = 0
	p=Kernel.shape[0]
	for i in range(0,p):
		s +=lambda_star[i]*y_train[i]*Kernel[i,idx]
	return s
	
def compute_b(x_train,y_train,lambda_star,gamma): 
	idx=np.where(lambda_star.round(9) != 0)[0][0] 
	k_pt=gaussian_kernel(x_train,x_train[idx].reshape(1,27),gamma).reshape(-1)
	y_pt=y_train[idx]
	elem=[lambda_star,y_train,k_pt]
	return (1/y_pt)-np.sum(reduce(np.multiply, elem))
		
def pred(x_pred,x_train,y_train,lambda_star,gamma):
	Kernel=gaussian_kernel(x_train,x_pred,gamma)
	w_star=np.zeros((x_pred.shape[0]))
	for idx in range(x_pred.shape[0]):
		w_star[idx] = compute_w(idx,lambda_star,y_train,Kernel)
	
	b_star = compute_b(x_train,y_train,lambda_star,gamma)
	res = w_star  + b_star
	y_pred=np.sign(res)
	return y_pred

def compute_accuracy(y_true, y_pred):
    return (accuracy_score(y_true,y_pred))
	
def jacobian(x,Q):
    c = np.repeat(-1,Q.shape[0])
    return np.dot(x.T, Q) + c

def find_newalpha(i,j,d,alpha,C,grad,Q):
	Qstar=np.zeros((2,2))
	W=[i,j]
	for c in range(2):
		for v in range(2):
			Qstar[c,v]=Q[W[c],W[c]]
	alpha_=alpha[[i,j]]
	d_plus = d[W]
	grad_MVP = np.array([grad[i], grad[j]])
	grad_d = np.dot(grad_MVP.T, d_plus)
	if np.isclose(grad_d, 0):
		alpha_star = alpha_
		print('beta_star is useless')
		return alpha_star
	elif grad_d < 0: d_star = d_plus
	elif grad_d > 0: d_star = -d_plus
	
	if d_star[0]>0 and d_star[1]>0:
		beta=min(C-alpha_[0],C-alpha_[1])
	if d_star[0]<0 and d_star[1]<0:
		beta=min(alpha_[0],alpha_[1])
	if d_star[0]>0 and d_star[1]<0:
		beta=min(C-alpha_[0],alpha_[1])
	if d_star[0]<0 and d_star[1]>0:
		beta=min(alpha_[0],C-alpha_[1])
	if np.isclose(beta, 0):
		alpha_star = alpha_
		print('beta is: ',beta)
		return alpha_star
	if np.isclose(np.dot(d_star.T,Qstar).dot(d_star),0):
		beta_star=beta
	else:
		beta_nv=(-np.dot(grad_MVP.T,d_star))/(np.dot(d_star.T,Qstar).dot(d_star))
		beta_star=min(beta_nv,beta)
	print('beta_star founded is: ',beta_star)
	alpha_star=alpha_+ beta_star*d_star
	return alpha_star
	
def create_R_S(lambda_val,C,y):
	train_index = set(np.arange(lambda_val.shape[0]))
	L = set([i for i in train_index if np.isclose(lambda_val[i],0)])
	U = set([i for i in train_index if np.isclose(lambda_val[i],C)])
	F = set([i for i in train_index if  lambda_val[i] > 0 and lambda_val[i] < C])
	L_positive = set([i for i in L if y[i] > 0])
	L_negative = set([i for i in L if y[i] < 0])
	U_positive = set([i for i in U if y[i] > 0])
	U_negative = set([i for i in U if y[i] < 0])
	R_set = list(L_positive.union(U_negative).union(F))
	S_set = list(L_negative.union(U_positive).union(F))
	return R_set,S_set
	
def WSS(lambda_values,C,y_train,fraction,q):
	R,S=create_R_S(lambda_values,C,y_train)
	R_values = [ round(elem, 6) for elem in fraction[R] ]
	S_values = [ round(elem, 6) for elem in fraction[S] ]
	I=[g for _,g in sorted(zip(R_values,R),key = lambda x: x[0],reverse=True)]
	J=[x for _,x in sorted(zip(S_values,S),key = lambda x: x[0],reverse=False)]
	m = fraction[I[0]]
	M = fraction[J[0]]
	return I[0:int(q/2)],J[0:int(q/2)],m,M
	
def function_to_minimize_QP_DM(x,x_fix,Q_WW,Q_WbarW):
	c = np.repeat(1,Q_WW.shape[0]).reshape(1,-1)
	elem1=[x.T,Q_WW,x]
	elem2=[x_fix.reshape(1,-1),Q_WbarW]
	obj = 0.5 * reduce(np.dot, elem1) + np.dot((reduce(np.dot, elem2) - c.reshape(1,-1) ) , x.reshape(-1,1) )
	return obj[0][0]

# Jacobian (gradient) of objective function. 
def jacobian_QP_DM(x,x_fix,Q_WW,Q_WbarW):
	c = np.repeat(1,Q_WW.shape[0]).reshape(1,-1)
	return np.dot(x.reshape(1,-1), Q_WW) + np.dot(x_fix.reshape(1,-1),Q_WbarW) - c

def constraint(x,x_fix,W,W_bar,y):
	return np.inner(y[W],x) + np.inner(y[W_bar],x_fix)

def QP_minimization_DM(Q_RBF,i,j,q,lambda_values,y,C,tol=1e-6):
	
	p=Q_RBF.shape[0]
	W=i+j	
	W_bar=np.delete(list(range(p)),W).tolist()
	x_fix=lambda_values[W_bar]
	Q_WW=np.zeros((q,q))	
	for m in range(q):
		for l in range(q):
			Q_WW[m,l] = Q_RBF[W[m],W[l]]
	Q_WbarW=np.zeros((p-q,q))
	for t in range(p-q):
		for r in range(q):
			Q_WbarW[t,r] = Q_RBF[W_bar[t],W[r]]

	cons = ({'type':'eq', 'fun':lambda t: constraint(t,x_fix,W,W_bar,y)})

	x = np.repeat(0,q)
	bound = tuple((0,C) for i in range(q))
	res_SLSQP = minimize(fun = function_to_minimize_QP_DM, 
							x0 = x, 
							args= (x_fix,Q_WW,Q_WbarW),
							jac = jacobian_QP_DM ,
							constraints = cons, 
							bounds = bound, 
							method ='SLSQP', options={'ftol': tol, 'disp': True})
	print(res_SLSQP.message)

	return res_SLSQP
	
def decomposition_method(Q_RBF,y_train,C_DM,eps,q=2,max_iter=500):
	# Stopping Criteria	
	iteration = 0
	lambda_values = np.zeros(Q_RBF.shape[0])

	function_eval_list=[]
	gradient_eval_list = []

	gradient = -np.ones((Q_RBF.shape[0]))
	fraction = -np.multiply(gradient,y_train)
	
	startTime_total = time.time()
	while True:
		print ("# iter: ", iteration)
		if iteration == max_iter:
			print ("we have reached the maximum number of iterations")
			break
		# 1 step  
		i,j,m,M=WSS(lambda_values,C_DM,y_train,fraction,q)
		W=i+j
		print('index i selected: ',i,'index j selected: ',j)
		# 2 Step		
		res_SLSQP_reduced=QP_minimization_DM(Q_RBF,i,j,q,lambda_values,y=y_train,C=C_DM,tol=1e-6)
		
		function_eval_list.append(res_SLSQP_reduced.nfev)
		gradient_eval_list.append(res_SLSQP_reduced.njev)
		# 3 Step
		old_lambda=lambda_values.copy()
		for index,d in zip(W,range(q)):
			lambda_values[index] = res_SLSQP_reduced.x[d] 
		# 4 step 	
		gradient = jacobian_QP(lambda_values,Q_RBF)
		#gradient= gradient + (lambda_values)*sum([Q_RBF[i] for i in W ])
		fraction = -np.multiply(gradient,y_train)
		iteration +=1
		print('Value of m: ',m,'Value of M: ',M)
		print ("--------------------------------------------------------------")
		print()
		if m <= M + eps:
			print ("Optimality reached!!")
			break  
	print()
	endTime_total = time.time()
	timeElapsed_total = endTime_total - startTime_total
	print("Elapsed time total:", timeElapsed_total)
	print("Function evaluation total:", sum(function_eval_list))
	print("Gradient evaluation total:", sum(gradient_eval_list))
	return [lambda_values, [res_SLSQP_reduced, sum(function_eval_list), sum(gradient_eval_list), timeElapsed_total, iteration]]
		
def MVP_method(Q_RBF,y_train,C,max_iter):
	iteration = 0    
	lambda_values = np.zeros(Q_RBF.shape[0])

	gradient = -np.ones((Q_RBF.shape[0]))
	fraction = -np.multiply(gradient,y_train)
	
	startTime_total = time.time()
	while True:

		startTime = time.time()
		print ("# iter: ", iteration)
			   
		if iteration == max_iter:
			print ("we have reached the maximum number of iterations")
			break
		R,S=create_R_S(lambda_values,C,y_train)
		
		R_values =  [ round(elem, 5) for elem in fraction[R] ]
		S_values = [ round(elem, 5) for elem in fraction[S] ]
		I=[g for _,g in sorted(zip(R_values,R),key = lambda x: x[0],reverse=True)]
		J=[x for _,x in sorted(zip(S_values,S),key = lambda x: x[0],reverse=False)]
		i = I[0]
		j = J[0]
		print('index i selected: ',i,'index j selected: ',j)
		
		m = fraction[i]
		M = fraction[j]
		if m <= M + 0.5 :
			print ("Optimality reached!!")
			break
		
		print("Indexes",i,j,"selected are MVP")
		
		d=np.zeros(Q_RBF.shape[0])
		d[i]=1/y_train[i]
		d[j]=-(1/y_train[j])
		
		alpha_star=find_newalpha(i,j,d,lambda_values,C,gradient,Q_RBF)
		print('alpha_star founded is: ',alpha_star)
		lambda_values[[i,j]]=alpha_star[[0,1]]
		gradient= jacobian_QP(lambda_values,Q_RBF)
		fraction = -np.multiply(gradient,y_train)
		
		endTime = time.time()
		timeElapsed = endTime - startTime
		print ("Elapsed time:", timeElapsed)
		print()
		iteration +=1
		print ("--------------------------------------------------------------")
		print()
	print()   
	endTime_total = time.time()
	timeElapsed_total = endTime_total - startTime_total
	print("Elapsed time total:", timeElapsed_total)
	return [lambda_values, timeElapsed_total, iteration]