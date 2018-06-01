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

def softmax_(z):
    b = np.max(z,axis=0,keepdims=True)
    exp_z = np.exp(z-b)
    summed = np.sum(exp_z,axis=0,keepdims=True)
    return exp_z/summed

# softOut = K x 1 softmax output
def dSoftmax(softOut):
    k = np.size(softOut,0)
    # p_i is K x K matrix of repeated columns of softOut
    p_i = np.repeat(softOut,k,axis=1)
    # p_j : K x K mtx where every column corresponds to softOut at particular index (ie. the tranpose of p_i)
    p_j = p_i.T
    p_i_Diag = np.diagflat(softOut)
    deriv = p_i_Diag - p_i * p_j
    return deriv

def get_output_dW(allLayers,layerN,dE_dO):
    all_dE_dZ = []
    gradients = []
    tempOut = allLayers[layerN - 1]
    tempIn = allLayers[layerN - 2]
    for ind in range(np.size(allLayers[0], 1)):
        dSoft = dSoftmax(tempOut[:, np.newaxis, ind])
        temp_dEdZ = np.dot(dE_dO[:, ind,np.newaxis].T, dSoft)
        print(dE_dO[:,ind].shape)
        temp_dEdW = np.dot(tempIn[:, np.newaxis, ind], temp_dEdZ)
        all_dE_dZ.append(temp_dEdZ)
        gradients.append(temp_dEdW)
    return gradients, all_dE_dZ

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
    layerGrad = []
    gradients = []
    newList = [inpt] + outputs
    nlistLen = len(newList)
    o_ = newList[nLayer+1]
    dE_dO = o_ - t
    ##EE = -np.sum(np.dot(np.transpose(t),np.log(o_)))
    EE = np.sum(np.sum(0.5 * (dE_dO)**2,axis=1))
    print("EE ")
    print(EE)

    prevGrad = dE_dO

    for i in range(nlistLen - 1,0,-1):
        # PROBLEMS WITH O_K and X_K
        o_k = newList[i]
        x_k = newList[i - 1]
        dSig_O = dSig(o_k)

        dE_dZ = prevGrad * dSig_O
        print(prevGrad.shape)
        #dE_dZ = np.dot(prevGrad,dSig(o_k))

        dE_dW = np.matmul(x_k.T[:,:,np.newaxis],dE_dZ.T[:,np.newaxis,:])
        gradients.append(dE_dW)

        #UNCOMMENT IF NECESSARY
        # for ex in range(x_k.shape[1]):
        #     x_k_n = x_k[:, np.newaxis, ex]
        #     dE_dW = np.dot(x_k_n,dE_dZ[:,ex,np.newaxis].T)
        #     # dE_dW = dE_dZ * x_k
        #     #dE_dW_ = np.sum(dE_dW,axis=1,keepdims=True)
        #     if ex <= x_k.shape[1] / 2  and ex > 2:
        #         layerGrad[0] += dE_dW
        #     elif ex > (x_k.shape[1] / 2):
        #         layerGrad[1] += dE_dW
        #     else:
        #         layerGrad.append(dE_dW)

            #prevGrad = dE_dZ * w[i-1]
        prevGrad = np.dot(np.transpose(w[i-1]),dE_dZ)
            #prevGrad = np.dot(dE_dZ,w[i-1])

        #UNCOMMENT If NECESSARY
        #gradients.append(layerGrad)
        #layerGrad = []

    print("grad len " + str(gradients[1].shape))
    return gradients

def train(inpt,w,nHidLayers,t,f,g,numEpoch,learnR):
    for epoch in range(numEpoch):
        print(epoch)
        outputs = feed_forward(inpt, w, nHidLayers, f, g)
        gradients = back_propagate(inpt,w,nHidLayers,outputs,t)
        gradLen = len(gradients)
        for ind in range(gradLen-1,-1,-1):
            w_ind = gradLen - 1 - ind
            for n_grad in gradients[ind]:
                w[w_ind] -= (learnR * n_grad.T)
    return w

def predict(inpt,w,nLayers,f,g,t):
    outputs = feed_forward(inpt,w,nLayers,f,g)
    prediction = outputs[len(outputs) - 1]
    np.savetxt("Predictions.txt", prediction, delimiter=",")
    counter = 0
    hardPredict = np.argmax(prediction, axis=0)
    for ind,val in enumerate(hardPredict):
        if val == t[ind]:
            counter +=1
    print ("Test accuracy: " + str(float(counter)/len(hardPredict)))

if __name__=='__main__':
    nHidUnits = [11]
    nHidLay = len(nHidUnits)
    nOutputU = 10
    #w = []

    # x is a N x numInputs mtx
    x = np.loadtxt(open("/home/alex/ML_datasets/fashion-mnist_train.csv", "rb"), delimiter=",", skiprows=1)
    x_test = np.loadtxt(open("/home/alex/ML_datasets/fashion-mnist_test.csv", "rb"), delimiter=",", skiprows=1)
    #x = np.loadtxt(open("/home/alex/ML-models/train_proto.csv", "rb"), delimiter=",", skiprows=1)
    #x_test = np.loadtxt(open("/home/alex/ML-models/train_proto.csv", "rb"), delimiter=",", skiprows=1)
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
    w_trained = train(x,weights,nHidLay,t,sigmoid_,softmax_,700,0.1)
    #print(w_trained[0])
    for index,w_ in enumerate(w_trained):
        np.savetxt("weights_nn_layer" + str(index) + ".txt",w_,delimiter=",")
    # for index in range(nHidLay+1):
    #     weights.append(np.loadtxt(open("weights_nn_layer" + str(index) + ".txt","rb"),delimiter=","))
    predict(x_test,w_trained,nHidLay,sigmoid_pos,softmax_,truth_test)
