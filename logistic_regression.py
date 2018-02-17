import numpy as np

#Produces a normalized exponential output (softmax) for each class (K classes in total)
# Inputs
#   w = weights (K,W)
#   x = features (W,N)
#### Issue with overflow from Exponential of activation z ####

def prediction(w,x):
    z = np.dot(w,x)    # Activations
    numer = np.exp(z)
    denom = np.sum(numer, axis=0,keepdims=True)
    return numer/denom


#Performs weight optimization (Gradient Descent for this case)
# Inputs:
#   w = weights  (K,W)
#   x = features (W,N)
#   t = 1-of-k encoding data labels

def train(w,x,t):
    x = np.array(x,dtype=np.float128)
    n = x.shape[1]
    x_trans = np.transpose(x)

    for i in range(100):
        print(i)
        y = prediction(w, x)
        error = y - t

        dw = np.dot(error, x_trans) / n
        w -= 0.1 * dw
    return w

def get_moments(x):
    return np.mean(x, axis=1,keepdims=True), np.std(x,axis=1,keepdims=True)

def standardize(x,means=0,sd=1):
    return (x - means)/sd