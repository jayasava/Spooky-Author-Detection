import numpy as np
train_inputs=np.asarray([[0,0,0],[0,0,1],[1,0,1],[1,1,0],[1,1,1]])
train_outputs=np.asarray([[0,0,1,1,1]]).T # Transpose
np.random.seed(1)
w0=2*np.random.random((3,1))-1

def sigmoid(dot_prod):
    func=1/(1+np.exp(dot_prod))
    return func

def derivation(x):
    derv=x*(1-x)
    return derv
    
for i in range(100000):
##Forward Propogation
    dot_prod=np.dot(train_inputs,w0)
    cost_func=sigmoid(dot_prod)
    error=train_outputs-cost_func
    delta=error*derivation(cost_func)
    w0=w0+np.dot(train_inputs.T,delta)
    
    
print(cost_func)








