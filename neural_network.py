import numpy as np
import debug_tools as dbg

sigmoid_pos = lambda x: 1.0 / (1.0 + np.exp(-x))
sigmoid_neg = lambda x: np.exp(x) / (1.0 + np.exp(x))
tanh = lambda x: (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
reLU = lambda x: max(0,x)

dSig = lambda x: x *(1.0 - x)

def sigmoid_(z):
    entries = [[sigmoid_pos(x) if x >= 0 else sigmoid_neg(x) for x in row] for row in z]
    return np.array(entries)
        # if z >= 0:
        #     return sigmoid_pos(z)
        # else:
        #     return sigmoid_neg(z)

def softmax_(z):
    b = np.max(z,axis=0,keepdims=True)
    exp_z = np.exp(z-b)
    summed = np.sum(exp_z,axis=0,keepdims=True)
    return exp_z/summed

# softOut = K x 1 softmax output
def dSoftmax(softOut):
    k = np.size(softOut,0)
    p_i = np.zeros([k,k])
    return temp

def get_moments(x):
    return np.mean(x, axis=1,keepdims=True), np.std(x,axis=1,keepdims=True)

def standardize(x,means=0,sd=1):
    return (x - means)/sd

# Randomly initialize list of weights for feedforward and updating
# Inputs:
#   nHidUnits = Number of units per layer
#   nOutU = Number of output units
def initWeights(nHidUnits, nIn, nOutU):
    axis1 = nIn
    w = []
    for outUnits in nHidUnits + [nOutU]:
        tempW = np.random.random((outUnits, axis1)) / 100
        w.append(tempW)
        axis1 = outUnits
    return w

# Pushes input through a layer
# Inputs:
#   inpt = K x N matrix of input vectors
#   w = J x K weight matrix
#   f = activation function for hidden layers
def feed_forward(inpt,w,nLayers,f,g):
    x = inpt.astype(dtype=np.float32)
    outputs = []
    for i in range(nLayers):
        w_curr = w[i]
        temp = np.dot(w_curr,x)
        x = f(temp)
        outputs.append(x)

    if nLayers > 0:
        pre = np.dot(w[nLayers],outputs[nLayers-1])
    else:
        pre = np.dot(w[nLayers],x)
    out_final = g(pre)
    return outputs + [out_final]

# Performs weight updates using a Squared Loss function, sigmoid activation and gradient descent
# Inputs:
#   inpt = K x N matrix of input vectors
#   w = J x K weight matrix for particular layer
#   prevGrad = Partial derivative of loss function with respect to the unit's pre-activation value (eg. dE_dz)
def back_propagate(inpt,w,nLayer,outputs,t):
    newList = [inpt] + outputs
    o_ = newList[nLayer+1]
    prevGrad = (o_ - t)
    # logg = np.log(o_)
    # transposed = np.transpose(t)
    # tempp = np.dot(transposed,logg)
    # EE  = -np.sum(tempp)
    ##EE = -np.sum(np.dot(np.transpose(t),np.log(o_)))
    EE = np.sum(np.sum(0.5 * (prevGrad)**2,axis=1))
    print("EE ")
    print(EE)

    gradients = []
    for i in range(len(newList) - 1,0,-1):
# PROBLEMS WITH O_K and X_K
        o_k = newList[i]
        x_k = newList[i-1]

        dE_dZ = prevGrad * dSig(o_k)
        #dE_dZ = np.dot(prevGrad,dSig(o_k))

        #dE_dW = dE_dZ * x_k
        #dE_dW = np.dot(dE_dZ,np.transpose(x_k))
        dE_dW = np.dot(dE_dZ,np.transpose(x_k))
        #dE_dW_ = np.sum(dE_dW,axis=1,keepdims=True)
        gradients.append(dE_dW)

        #prevGrad = dE_dZ * w[i-1]
        prevGrad = np.dot(np.transpose(w[i-1]),dE_dZ)
        #prevGrad = np.dot(dE_dZ,w[i-1])

    return gradients

def train(inpt,w,nHidLayers,t,f,g,numEpoch,learnR):
    for epoch in range(numEpoch):
        print(epoch)
        outputs = feed_forward(inpt, w, nHidLayers, f, g)
        gradients = back_propagate(inpt,w,nHidLayers,outputs,t)
        for ind in range(len(gradients)-1,-1,-1):
            w_ind = len(gradients)- 1 - ind
            #print(w[w_ind])
            w[w_ind] -= (learnR * gradients[ind])
            #print(w[w_ind])
    return w

def predict(inpt,w,nLayers,f,g,t):
    outputs = feed_forward(inpt,w,nLayers,f,g)
    prediction = outputs[len(outputs) - 1]
    #print("prediciton")
    #print(prediction[:,4])
    np.savetxt("Predictions.txt", prediction, delimiter=",")
    counter = 0
    hardPredict = np.argmax(prediction, axis=0)
    for ind,val in enumerate(hardPredict):
        if val == t[ind]:
            counter +=1
    print ("Test accuracy: " + str(float(counter)/len(hardPredict)))

if __name__=='__main__':
    nHidUnits = [10]
    nHidLay = len(nHidUnits)
    nOutputU = 10
    #w = []

    # x is a N x numInputs mtx

    truth = np.transpose(x[:, 0]).astype(int)
    truth_test = np.transpose(x_test[:, 0]).astype(int)

    # Eliminate column of labels from set and reshape
    x = np.delete(x, np.s_[:1], 1)
    x = np.transpose(x)

    print(x.shape)
    nInputs = x.shape[0]

    # means,sd = get_moments(x)
    # x = standardize(x,means,sd)
    x = x/255

    x_test = np.delete(x_test, np.s_[:1], 1)
    x_test = np.transpose(x_test)

    x_dim = x.shape
    t = np.zeros((10, x_dim[1]))
    arange = np.arange(x_dim[1])
    t[truth, arange] = 1
    print(t)

    #weights = []
    weights = initWeights(nHidUnits,nInputs,nOutputU)


    #print(weights[0])
    w_trained = train(x,weights,nHidLay,t,sigmoid_,softmax_,1000,0.1)
    #print(w_trained[0])
    for index,w_ in enumerate(w_trained):
        np.savetxt("weights_nn_layer" + str(index) + ".txt",w_,delimiter=",")
    # for index in range(nHidLay+1):
    #     weights.append(np.loadtxt(open("weights_nn_layer" + str(index) + ".txt","rb"),delimiter=","))
    predict(x_test,w_trained,nHidLay,sigmoid_pos,softmax_,truth_test)