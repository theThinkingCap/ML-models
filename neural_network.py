import numpy as np
import debug_tools as dbg

sigmoid_pos = lambda x: 1.0 / (1.0 + np.exp(-x))
sigmoid_neg = lambda x: np.exp(x) / (1.0 + np.exp(x))
tanh = lambda x: (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
reLU = lambda x: max(0,x)

dSig = lambda x: x *(1.0 - x)



class NeuralNet:

    # Members:
    #   hUPL = Number of hidden units per layer
    #   nHL = Number of hidden layers
    #   nIn = Number of units in input data
    #   nOutU = Number of output units
    #   w = (nHL + 1) list of J x K weight matrix
    def __init__(self,nHidUnits,nIn,nOutU):
        self.hUPL = nHidUnits
        self.nHL = len(nHidUnits)
        self.nIn = nIn
        self.nOutU = nOutU
        self.w = self.initWeights()

    # Randomly initialize each weight layer
    def initWeights(self):
        axis1 = self.nIn
        w = []
        for outUnits in self.hUPL + [self.nOutU]:
            tempW = np.random.random((outUnits, axis1))
            w.append(tempW)
            axis1 = outUnits
        return w

    # Pushes input through a layer
    # Inputs:
    #   inpt = K x N matrix of input vectors
    #   f = activation function for hidden layers
    def feed_forward(self,inpt,f,g):
        x = inpt
        outputs = []
        for i in range(self.nHL):
            w_curr = self.w[i]
            temp = np.dot(w_curr,x)
            x = f(temp)
            outputs.append(x)

        if self.nHL > 0:
            pre = np.dot(self.w[self.nHL], outputs[self.nHL - 1])
        else:
            pre = np.dot(self.w[self.nHL], x)
        out_final = g(pre)
        return outputs + [out_final]

    # Performs weight updates using a Squared Loss function, sigmoid activation and gradient descent
    # Inputs:
    #   inpt = K x N matrix of input vectors
    #   prevGrad = Partial derivative of loss function with respect to the unit's pre-activation value (eg. dE_dz)
    def back_propagate(self,inpt,outputs,t):
        gradients = []
        newList = [inpt] + outputs
        nlistLen = len(newList)
        o_ = newList[self.nHL + 1]
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
            #dE_dZ = np.dot(prevGrad,dSig(o_k))

            dE_dW = np.sum(np.matmul(x_k.T[:,:,np.newaxis],dE_dZ.T[:,np.newaxis,:]),axis=0,keepdims=True)
            print("DE_DW SHAPE " + str(dE_dW.shape))
            gradients.append(dE_dW)

            prevGrad = np.dot(np.transpose(self.w[i-1]),dE_dZ)
        return gradients

    def train(self,inpt,t,f,g,numEpoch,learnR):
        for epoch in range(numEpoch):
            print(epoch)
            outputs = self.feed_forward(inpt, f, g)
            gradients = self.back_propagate(inpt,outputs,t)
            gradLen = len(gradients)
            for ind in range(gradLen-1,-1,-1):
                w_ind = gradLen - 1 - ind
                for n_grad in gradients[ind]:
                    #print("w shape " + str(w[w_ind].shape))
                    #print("n grad shape " + str(n_grad.T.shape))
                    self.w[w_ind] -= (learnR * n_grad.T)

    def predict(self,inpt,f,g,t):
        outputs = self.feed_forward(inpt,f,g)
        prediction = outputs[len(outputs) - 1]
        np.savetxt("Predictions.txt", prediction, delimiter=",")
        counter = 0
        hardPredict = np.argmax(prediction, axis=0)
        for ind,val in enumerate(hardPredict):
            if val == t[ind]:
                counter +=1
        print ("Test accuracy: " + str(float(counter)/len(hardPredict)))

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

def get_moments(x):
    return np.mean(x, axis=1,keepdims=True), np.std(x,axis=1,keepdims=True)

def standardize(x,means=0,sd=1):
    return (x - means)/sd