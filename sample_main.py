import numpy as np
import neural_network as NN

##########################################################
##Currently running variable layer neural network model##
##########################################################

if __name__=='__main__':
    nHidUnits = []
    nHidLay = len(nHidUnits)
    nOutputU = 10

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

    x = x.astype(float)
    means,sd = NN.get_moments(x)
    x = NN.standardize(x,means,sd)

    #x = x/255

    x_test = np.delete(x_test, np.s_[:1], 1)
    x_test = np.transpose(x_test)

    x_dim = x.shape
    t = np.zeros((10, x_dim[1]))
    arange = np.arange(x_dim[1])
    t[truth, arange] = 1
    print(t)

    nn = NN.NeuralNet(nHidUnits,nInputs,nOutputU)

    # weights = initWeights(nHidUnits,nInputs,nOutputU)
    # w_trained = train(x,weights,nHidLay,t,sigmoid_,softmax_,600,0.1)
    # predict(x_test, w_trained, nHidLay, sigmoid_pos, softmax_, truth_test)

    #print(weights[0])
    for seg in range(1,9):
        startI = int((seg-1)*0.2*x_dim[1])
        endI = int(seg*0.2*x_dim[1])
        trainX = np.concatenate((x[:,:startI],x[:,endI:]),axis=1)
        trainTruth = np.concatenate((t[:,:startI],t[:,endI:]),axis=1)

        nn.train(trainX,trainTruth,NN.sigmoid_,NN.softmax_,400,0.1)
        #print(w_trained[0])

        # for index,w_ in enumerate(w_trained):
        #     np.savetxt("weights_nn_layer" + str(index) + ".txt",w_,delimiter=",")

        # for index in range(nHidLay+1):
        #     weights.append(np.loadtxt(open("weights_nn_layer" + str(index) + ".txt","rb"),delimiter=","))
        nn.predict(x_test,NN.sigmoid_pos,NN.softmax_,truth_test)