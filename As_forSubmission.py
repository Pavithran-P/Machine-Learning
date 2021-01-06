# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:42:17 2020

for submission

@author: Jarvis
"""

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Loading data
# =============================================================================
def loadData():
    with np.load('notMNIST.npz') as data:
        Data, Target = data['images'], data['labels']
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
       
    return trainData, validData, testData, trainTarget, validTarget, testTarget
# =============================================================================
# Data Processing
# =============================================================================
    
# =============================================================================
#     1. Theory
# =============================================================================

# =============================================================================
#     2. One hot
# =============================================================================

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData() 


# =============================================================================

 
def convertOneHot(trainTarget, validTarget, testTarget):
    #function for converting to one hot
    def one_hot(targets, n):
        out = np.eye(n)[np.array(targets).reshape(-1)]
        return out.reshape(list(targets.shape)+[n])
    trainTarget_onehot = one_hot(trainTarget,10)
    validTarget_onehot = one_hot(validTarget,10)
    testTarget_onehot = one_hot(testTarget,10)
    return trainTarget_onehot, validTarget_onehot, testTarget_onehot



trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
# =============================================================================
#     3. Structure of the network  (Diagram)
# =============================================================================
    
# =============================================================================
#  4. Helper functions
# =============================================================================


def relu(x):
    return np.maximum(x, 0)

def softmax(x):       
    exp = np.exp(x)
    return exp/np.sum(exp, axis = 1, keepdims = True)

def computeLayer(x,W):

    s = np.dot(x,W)
    return s

def CE(target, prediction):
    epsilon=1e-22
    prediction = softmax(prediction)
    prediction = np.clip(prediction, epsilon, 1. - epsilon)
    N = prediction.shape[0]
    ce = -np.sum(target*np.log(prediction+1e-22))/N
    return ce

def gradCE(target, prediction):
    grad = softmax(prediction) - target
    return grad


def accuracy(x, y, wh, wob, wo):
    accCurve = []
    
    input_layer = np.dot(x, wh)
    hidden_layer = relu(input_layer)
    scores = np.dot(hidden_layer, wo) + wob
    probs = softmax(scores)
    
    correct = 0
    total = 0
    for i in range(len(y)):
        act_label = np.argmax(y[i]) # act_label = 1 (index)
        pred_label = np.argmax(probs[i]) # pred_label = 1 (index)
        if(act_label == pred_label):
            correct += 1
        total += 1
        accCurve.append((correct/total)*100)
    accuracy = (correct/total)*100
#    error = CE(testTarget,scores)
    error = 0
    return accuracy, accCurve, error

def testCurves(x, y, wh, wob, wo):
    accCurve = []
    
    input_layer = np.dot(x, wh)
    hidden_layer = relu(input_layer)
    scores = np.dot(hidden_layer, wo) + wob
    probs = softmax(scores)
    
    correct = 0
    total = 0
    for i in range(len(x)):
        act_label = np.argmax(y[i]) # act_label = 1 (index)
        pred_label = np.argmax(probs[i]) # pred_label = 1 (index)
        if(act_label == pred_label):
            correct += 1
        total += 1
        accCurve.append((correct/total)*100)
    accuracy = (correct/total)*100
#    print('accuracy = ', accuracy)
#    e = CE(testTarget,scores)
#    print('\tError = ',e)
    return accuracy, accCurve
    

def Train(trainData, trainTarget, wh, wob ,wo, epochs, eta, momentum,
          validData, validTarget, testData, testTarget):
    
# =============================================================================
#     Init
# =============================================================================
    
    vh, prevh = np.ones(wh.shape), np.ones(wh.shape)
    vbo, prevbo = np.ones(wob.shape), np.ones(wob.shape)
    vo, prevo = np.ones(wo.shape), np.ones(wo.shape)
    
    vh, prevh = np.multiply(vh,1e-5), np.multiply(vh,1e-5)
    vbo, prevbo = np.multiply(vbo,1e-5), np.multiply(vbo,1e-5)
    vo, prevo = np.multiply(vo,1e-5), np.multiply(vo,1e-5)
    
# =============================================================================
#     Function to find classification accuracy
# =============================================================================

# =============================================================================
#     Backpropagaion algorithm
# =============================================================================
    batch_size = 1000
    loss = []
    accu = []
    print('*********Training has begun*********')
    for epoch in range(epochs): # training begin
        iteration = 0
        while iteration < len(trainData):
            # batch input
            inputs_batch = trainData[iteration:iteration + batch_size]
            labels_batch = trainTarget[iteration:iteration + batch_size]
            
            # forward pass
            sh = computeLayer(inputs_batch,wh)
            xh = relu(sh)
            so = computeLayer(xh,wo) + wob
            xo = softmax(so)
        
            # calculate loss
            loss1 = CE(labels_batch,so)
            loss.append(loss1)

            # backward pass

#            delta_y = (xo - labels_batch)  / xo.shape[0]
#            delta_y = gradCE(labels_batch,so)  / so.shape[0]
            delta_y = gradCE(labels_batch,so) #derivatives of softmax layer(output)
            delta_hidden_layer = np.dot(delta_y, wo.T) 
            delta_hidden_layer[xh <= 0] = 0 # derivatives of relu
            
            # backpropagation
            weight2_gradient = np.dot(xh.T, delta_y) # output layer
            bias2_gradient = np.sum(delta_y, axis = 0, keepdims = True) #output bias
            weight1_gradient = np.dot(inputs_batch.T, delta_hidden_layer) #hidden layer


            #Update the velocity vectors            
            vh = (momentum*prevh) -  (eta * weight1_gradient)
            vbo = (momentum*prevbo) -  (eta * bias2_gradient)
            vo = (momentum*prevo) -  (eta * weight2_gradient)

            #Update weight vectors
            wh = wh + vh
            wob = wob + vbo
            wo = wo + vo
            
            prevh = vh;
            prevbo = vbo
            prevo = vo
            
            iteration += batch_size
            
        TrainingAccuracy, curve, e = accuracy(trainData, trainTarget, wh, wob, wo)
        accu.append(TrainingAccuracy)
        print('===> Epoch:', epoch+1,'/',epochs, '\tLoss:' , loss1, ' Accuracy:', TrainingAccuracy)        
    print('******Training done******')    
    plt.figure()    
    plt.plot(loss)
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cross entropy loss')
    plt.figure()
    plt.plot(accu)
    plt.title('Classification Accuracy for Training data')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    

    
    return wh, wob, wo


def NeuralNetTrain(trainData,trainTarget,validData, validTarget, testData, 
                   testTarget, HiddenNeurons,epochs,eta,momentum):
    #init
    def Reshape(z):        
        a = []
        for i in range(len(trainData)):
            trainData[i] = np.array(trainData[i])
            b = trainData[i].flatten()
            b = [1,*b]
            b = np.asarray(b)
            a.append(b)
        a = np.asarray(a)
        return a
    
    trainData_reshaped = Reshape(trainData)
    validData_reshaped = Reshape(validData)
    testData_reshaped = Reshape(testData)
    
    #Xavier initialization
    mu = 0 #iid with zero mean
    s1 = np.sqrt(2 / (785 + HiddenNeurons)) # sigma^2 = (2 / (#inputNodes + #outputNodes)
    wh = np.random.normal(mu, s1, [785, HiddenNeurons]) #Gaussian distribution
    s2 = np.sqrt(2 / (HiddenNeurons + 10)) # sigma^2 = (2 / (#inputNodes + #outputNodes)
    wo = np.random.normal(mu, s2, [HiddenNeurons, 10])
    wob = np.zeros((1, 10))   


    wh, wob, wo = Train(trainData_reshaped, trainTarget, wh, wob ,wo, epochs, eta, momentum,
          validData_reshaped, validTarget, testData_reshaped, testTarget)
    
    return wh, wob, wo
    



wh, wob, wo = NeuralNetTrain(trainData,trainTarget,validData, validTarget, testData, 
                   testTarget,1000,100,10e-5,0.9)       



def testCurve(testData, testTarget, wh, wob, wo):
    accCurve = []
    a = []
    for i in range(len(testData)):
        testData[i] = np.array(testData[i])
        b = testData[i].flatten()
        b = [1,*b]
        b = np.asarray(b)
        a.append(b)
    a = np.asarray(a)
    
    input_layer = np.dot(a, wh)
    hidden_layer = relu(input_layer)
    scores = np.dot(hidden_layer, wo) + wob
    probs = softmax(scores)
    
    correct = 0
    total = 0
    for i in range(len(testData)):
        act_label = np.argmax(testTarget[i]) # act_label = 1 (index)
        pred_label = np.argmax(probs[i]) # pred_label = 1 (index)
        if(act_label == pred_label):
            correct += 1
        total += 1
        accCurve.append((correct/total)*100)
    accuracy = (correct/total)*100
#    print('accuracy = ', accuracy)
    e = CE(testTarget,scores)
#    print('\tError = ',e)
    return accuracy, accCurve,e          
    

# =============================================================================
# Validation accuracy
# =============================================================================

print('*****************************Validation********************************\n')
ValidationAccuracy,ValCurve,e1 = testCurve(validData, validTarget, wh, wob, wo)
print('Validation Accuracy = ',ValidationAccuracy,'\tLoss = ',e1)
plt.figure()
plt.plot(ValCurve)
plt.title("Validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Classification Accuracy for Validation data")
plt.show() 

# =============================================================================
# Test accuracy
# =============================================================================

print('********************************Test********************************\n')  
TestAccuracy,testAccCurve,e2 = testCurve(testData, testTarget, wh, wob, wo)
print('Test Accuracy = ',TestAccuracy,'\tLoss = ',e2)
plt.figure()
plt.plot(testAccCurve)
plt.title("Test accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy') 
plt.title("Classification Accuracy for Test data")
plt.show()


















