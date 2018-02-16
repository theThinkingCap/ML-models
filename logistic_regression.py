import numpy as np

#Produces a normalized exponential output (softmax) for each class

def prediction(w,x):
    z = np.dot(w,x)    # Activations
    numer = np.exp(z)
    denom = np.sum(numer, axis=0,keepdims=True)
    return numer/denom


#Performs weight optimization (Gradient Descent for this case)
# Inputs:
#   w = weights
#   x = features
#   t = 1-of-k encoding data labels

def train(w,x,t):
    #while error >= 10:
    for i in range(100):
        print(i)
        y = prediction(w, x)
        print("y shape")
        print(y.shape)
        print("t shape")
        print(t.shape)
        error = y - t

        #print("error shape")
        #print(error.shape)
        dw = np.dot(error, np.transpose(x))
        # dw is actually sum of (error * x_of_jth weight) for every N training example
        #x_copy = np.array()
        #dw = np.sum(dotted, axis=1)
        w += 0.1 * dw
    return w

def standardize(x):
    means = np.mean(x, axis=0,keepdims=True)
    sd = np.std(x,axis=0,keepdims=True)
    print (x - means)/sd
    return (x - means)/sd