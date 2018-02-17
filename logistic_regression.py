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

def standardize(x):
    means = np.mean(x, axis=1,keepdims=True)
    sd = np.std(x,axis=1,keepdims=True)
    return (x - means)/sd