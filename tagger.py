import sys
import os
import csv
import math
import numpy as np
from itertools import chain
import copy

def parsing(input, mode):
    #print("in parsing")
    file = list()
    with open(input,'r') as tsv:
        #print("get file")
        for line in csv.reader(tsv, delimiter="\t"):
            file.append(line)
        #print("file ready")

    #print(file)
    result = handleData(file, mode)
    return result

def run(train_input,val_input,test_input,train_out,test_out,metrics,num,mode):
    #print("in run")
    # get training data
    input = parsing(train_input, mode)
    allWords = input[0]
    allTags = input[1]
    feats = input[2]
    labels = input[3]
    labelsWithGap = input[4]
    #print("allwords: ",allWords)
    #print("get training")
    
    # get validation data
    valData = parsing(val_input, mode)
    valWords = copy.deepcopy(allWords)
    valFeats = valData[2]
    valLabels = valData[3]
    valMatrix = getXMatrix(valWords, valFeats, mode)
    #print("get validation")
    
    # set up matrices
    trainWords = copy.deepcopy(allWords)
    xMatrix = getXMatrix(trainWords, feats, mode)
    #print("trainWords", trainWords)
    #print("length: ", len(trainWords))
    #width = len(allWords) + 1
    if (mode == 1):
        width = len(trainWords) + 1
    else:
        width = len(trainWords) * 3 + 1
    height = len(allTags)
    thetaMatrix = getThetaMatrix(width, height)
    #for theta in thetaMatrix:
        #print(theta)
    #print("set up matrix")
    
    # compute theta
    res = updateTheta(xMatrix, valMatrix, thetaMatrix, allTags, 
                      labels, valLabels, num)
    theta = res[0]
    likelihood = res[1]  #likelihood for train and va data
    #print("get theta")

    # output for train data
    trainRes = test(theta, xMatrix, allTags, labelsWithGap)
    train_str = trainRes[0]
    train_err = trainRes[1]
    trainFile = open(train_out, 'w')
    trainFile.write(train_str)
    trainFile.close()
    #print("test trainning")
    
    # output for test data
    testData = parsing(test_input, mode)
    testWords = copy.deepcopy(allWords)
    testFeats = testData[2]
    testLabels = testData[4]
    testMatrix = getXMatrix(testWords, testFeats, mode)

    testRes = test(theta, testMatrix, allTags, testLabels)
    test_str = testRes[0]
    test_err = testRes[1]
    testFile = open(test_out, 'w')
    testFile.write(test_str)
    testFile.close()
    #print("test test")
    
    err_train = "error(train): {}\n".format(train_err)
    err_test = "error(test): {}\n".format(test_err)
    likelihood += err_train
    likelihood += err_test
    erroFile = open(metrics, 'w')
    erroFile.write(likelihood)
    erroFile.close()

##
## Helper function   
# handleData takes in 2d list data, handle it according to feature_flag
# output: list of all words(sorted), list of all labels,
#         1d list of features, 1d list of labels
def handleData(file, feature):
    words = list()   #collect all words
    tags = list()    #collect all labels
    feats = list()
    sample_feats = list()
    labels = list()
    labelsWithGap = list()
    sample_labels = list()
    #print("handling data")
    for sample in file:
        if (sample == []):
            if (feature == 2):
                sample_feats = modifyFeats(sample_feats)
            #print("sample_feats: ",sample_feats)
            labels.append(sample_labels)
            labelsWithGap.append(sample_labels)
            labelsWithGap.append([[]])
            feats.append(sample_feats)
            sample_labels = []
            sample_feats = []
        else:
            sample_feats.append(sample[0])
            sample_labels.append(sample[1])
            words.append(sample[0])
            tags.append(sample[1])
    if (feature == 2):
        sample_feats = modifyFeats(sample_feats)
    labels.append(sample_labels)   #last sample
    feats.append(sample_feats)     #last sample
    labelsWithGap.append(sample_labels)

    
    allWords = sorted(list(set(words)))
    allTags = sorted(list(set(tags)))
    
    #print(allWords)
    #print(allTags)
    feats = list(chain.from_iterable(feats))
    labels = list(chain.from_iterable(labels))
    labelsWithGap = list(chain.from_iterable(labelsWithGap))
    #print(feats)
    #print(labels)
    
    return [allWords, allTags, feats, labels, labelsWithGap]

def modifyFeats(feats):
    l = len(feats) + 1
    feats.insert(0, "BOS")
    feats.append("EOS")
    new = []
    
    for i in range(1,l):
        new.append([feats[i-1], feats[i], feats[i+1]])
    
    return new

def getXMatrix(allWords, feats, mode):
    if (mode == 2):
        allWords.insert(0, "BOS")
        allWords.append("EOS")

    width = len(allWords)
    matrix = list()
    #print(feats)
    #print(allWords)
    
    if (mode == 1):
        for word in feats:
            #print("word",word)
            row = [] 
            index = allWords.index(word)
            row.append(index)          # append the position of feat
            #print(row)
            matrix.append(row)
    else:
        for word in feats:
            #print(word)
            row = [] 
            index1 = allWords.index(word[0])
            index2 = allWords.index(word[1]) + width
            index3 = allWords.index(word[2]) + 2 * width
            row.append(index1)
            row.append(index2)
            row.append(index3)
            #print(row)
            matrix.append(row)
    #print(matrix)
    return matrix

def getThetaMatrix(width, height):
    matrix = []
    for i in range(height):
        row = [0] * width
        #row = np.array(row)
        matrix.append(row)
    return matrix

def updateTheta(xMatrix,valMatrix,thetaMatrix,allTags,labels,valLabels,num):
    theta = thetaMatrix
    str = ""
    for i in range(num):
        result = update(xMatrix, valMatrix, theta, allTags, labels, valLabels)
        theta = result[0]
        train_log = result[1]
        val_log = result[2]
        tempstr1 = "epoch={} likelihood(train): {}\n".format(i+1, train_log)
        tempstr2 = "epoch={} likelihood(validation): {}\n".format(i+1, val_log)
        str += tempstr1
        str += tempstr2
    return [theta, str]

def update(xMatrix, valMatrix, theta, allTags, labels, valLabels):
    l = len(theta[0])
    for i in range(len(labels)):
        y = labels[i]
        #x = np.array(xMatrix[i])
        x = xMatrix[i]
        grads = []
        
        for j in range(len(allTags)):  #comput gradiante for theta
            p = allTags[j]
            thetaRow = theta[j]
            bool =  int(y == p)
            #print("x: ", x)
            #print("theta: ", theta)
            prob = getProb(x, theta, thetaRow)
            scalar = prob-bool
            
            grad = [0] * (l)
            for index in x:
                grad[index] = scalar
            grad[-1] = scalar
            
            #grad = scalar * (np.array(newX))
            #grad.tolist()
            #print("grad: ", grad)
            grads.append(grad)
            
        for k in range(len(allTags)):  #update all theta
            for q in x:
                theta[k][q] -= 0.5 * grads[k][q]
            theta[k][-1] -= 0.5 * grads[k][-1]
    train_log = getLog(xMatrix, theta, labels, allTags)
    val_log = getLog(valMatrix, theta, valLabels, allTags)
    #print(train_log)
    #print(val_log)
    
    return [theta, train_log, val_log]

#note: vecX is np.array, row in theta is np.array
def getProb(vecX, theta, thetaRow):
    total = 0
    for i in range(len(theta)):
        inner = newDot(vecX, theta[i])
        total += math.exp(inner)
    up = newDot(vecX, thetaRow)
    prob = math.exp(up) / total
    return prob

#base on e
def getLog(xMatrix, theta, labels, allTags):
    res = 0
    for i in range(len(xMatrix)):
        #vecX = np.array(xMatrix[i])
        vecX = xMatrix[i]
        y = labels[i]
        k = allTags.index(y)
        rowK = theta[k]
        inner = newDot(vecX, rowK)
        up = math.exp(inner)
        down = 0
        for row in theta:
            inside = newDot(vecX, row)
            down += math.exp(inside)
        res += math.log(up / down)
    res /= len(labels)
    res *= -1
    return res

def test(theta, matrix, labels, labelsWithGap):
    #result = []
    erro = 0
    total = len(matrix)
    gap = 0
    str = ""
    for k in range(len(matrix)):
        #sample = np.array(matrix[k])
        sample = matrix[k]
        probs = []
        input = labelsWithGap[k + gap]
        
        if (input == []):
            gap += 1
            str += "\n"
        
        for row in theta:
            prob = newDot(sample, row)
            probs.append(prob)
        
        # find label with max likelihood
        arrayProbs = np.array(probs)
        indices = np.where(arrayProbs == max(probs))
        indices = indices[0].tolist()
        if (len(indices) == 1):
            index = indices[0]
            #result.append(labels[index])
            str += labels[index]
            if (labels[index] != labelsWithGap[k + gap]):
                erro += 1
        else:
            opt = []
            for i in indices:
                opt.append(labels[i])
            #result.append(min(opt))
            str += labels[index]
            if (labels[index] != labelsWithGap[k + gap]):
                erro += 1
        str += "\n"
    #print(str)
    rate = float(erro) / total
    #print(rate)
    return [str, rate]

def newDot(indices, vec):
    #print(indices)
    #print(vec)
    total = 0
    for i in indices:
        total += vec[i]
    total += vec[-1]
    return total
    

##
## Main function
if __name__ == '__main__':
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics = sys.argv[6]
    num_epoch = int(sys.argv[7])
    feature = int(sys.argv[8])
    
    run(train_input,validation_input,test_input,train_out,test_out,metrics,
        num_epoch,feature)