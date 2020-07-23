#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from dataloader import *
from pca import *
import random


# import data
images,labels = load_data()
emo_imgs = []
    
# parse the label of images and remove images labeled with 'neutral' or 'happy', key is subject
dict_label = {}
emotion = ['m','s','f','a','d']
for i in range(len(labels)):
    s = labels[i][:-4].split("_")
    if s[0] not in dict_label:
        dict_label[s[0]] = []
    if s[1][:2] == "ht" or s[1][0] in emotion:
        dict_label[s[0]] += [(s[1], images[i])]

# parse the label of images and remove images labeled with 'neutral' or 'happy', key is emotion
dict_label_ec = {}
emotion = ['m','s','f','a','d']
for i in range(len(labels)):
    s = labels[i][:-4].split("_")
    if s[1][:2] == "ht" or s[1][0] in emotion:
        if s[1][0] not in dict_label_ec:
            dict_label_ec[s[1][0]] = []
        dict_label_ec[s[1][0]] += [(s[0], images[i])]


# Logistic regression

# Choose images labeled e1 or e2 from a subject
# build its corresponding expected values
def subject_emo(e1,e2,subject,encode, dataset = dict_label):
    x_set = []
    t = []
    for i in subject:
        for l,img in dataset[i]:
            if l[0] == e1:
                x_set += [img]
                t += [encode]
      
            if l[0] == e2:
                x_set += [img]
                t += [int(not encode)]
            
    return (np.array(x_set),np.array(t))


# Choose a pair of subject as test set
# randomly choose a pair of subjects as holdout set from the rest
# then the remaining is train set
def choose_set(e1, e2, encode, index, dataset = dict_label):
    subject = list(dataset.keys())
    
    test = subject[2*index: 2*index+2]
    test_x, test_t = subject_emo(e1,e2,test,encode)
    
    rest = subject[:2*index] + subject[2*index+2:]
    random.shuffle(rest)
    
    holdout = rest[:2]
    holdout_x, holdout_t = subject_emo(e1,e2,holdout,encode)
    
    train = rest[2:]
    train_x, train_t = subject_emo(e1,e2,train,encode)
    
    return train_x,train_t,holdout_x, holdout_t,test_x, test_t


# Compute predicted values for logistic regression
def logistic_reg(w, x_N):
    y_N = np.array([1/(1+np.exp(-np.dot(w,x))) for x in x_N])
    return y_N


# Compute loss function for logistic regression
def loss_logistics(t_N, y_N):
    N = len(t_N)
    return sum([-(t_N[n]*np.log(y_N[n]) + (1-t_N[n])*np.log(1-y_N[n])) for n in range(N)])/N


# Compute the train error, holdout error and best weight vector for each run
def run_logistic(epoch,learning_rate,train_x,train_t,holdout_x,holdout_t,test_x,test_t):
    N = len(train_x)
    d = len(train_x[0])
    
    e_t = []
    e_h = []
    error = 1
    w_t = []

    # weight vector initialization
    w = np.zeros(d)
    w_t = np.zeros(d)
    
    # batch gradient descent for logistic regression
    for i in range(epoch):

        train_y = logistic_reg(w, train_x)
        e_t += [loss_logistics(train_t, train_y)]
        w = w + learning_rate/N*np.sum([np.dot(train_t[n]-train_y[n],train_x[n]) for n in range(N)], axis = 0) 
        
        # record error  
        holdout_y = logistic_reg(w, holdout_x)
        e_h += [loss_logistics(holdout_t, holdout_y)]

        # record best weights so far
        if e_h[-1] <= error:
            error = e_h[-1]
            w_t = w
    
    return e_t, e_h, w_t


# Plot train error, holdout error for 5 runs with different number of principal components
# Calculate corresponding test accuracy
def training_proc(epoch,kd,e1,e2,encode,learning_rate):
    e_train = {}
    e_holdout = {}
    e_test = {}

    for i in range(5):
        # choose train set, holdout set, test set 
        train_X,train_t,holdout_X, holdout_t,test_X, test_t = choose_set(e1,e2,encode,i)  
        
        for k_pca in kd:
            if k_pca not in e_train:
                e_train[k_pca] = []
                e_holdout[k_pca] = []
                e_test[k_pca] = []
                
            # transform images to k-dimension
            pca = PCA(k = k_pca)
            pca.fit(train_X)
            train_x = [j[0] for j in [pca.transform(i) for i in train_X]]
            holdout_x = [j[0] for j in [pca.transform(i) for i in holdout_X]]
            test_x = [j[0] for j in [pca.transform(i) for i in test_X]]

            e_t, e_h, w_t = run_logistic(epoch,learning_rate,train_x,train_t,holdout_x,holdout_t,test_x,test_t)

            # store train error, holdout error for each run
            e_train[k_pca] += [e_t]
            e_holdout[k_pca] += [e_h]
            
            # calculate test accuracy
            test_y = logistic_reg(w_t, test_x)
            e_test[k_pca] += [100*np.mean([(test_y > 0.5) == test_t])]
            
    for k in kd:
        plt.figure(kd.index(k))
        
        # calculate average of train error, holdout error for 10 epochs
        avg_train = [np.mean([x[i] for x in e_train[k]]) for i in range(10)]
        avg_holdout = [np.mean([x[i] for x in e_holdout[k]]) for i in range(10)]

        # calculate std of train error, holdout error for epoch 2,4,6,8
        std_train = [np.std([x[i-1] for x in e_train[k]]) for i in [2,4,6,8]]
        std_holdout = [np.std([x[i-1] for x in e_holdout[k]]) for i in [2,4,6,8]]
        
        # plot graphs
        x = [i for i in range(1,11)]
        plt.plot(x, avg_train, label = "avg train error")
        plt.plot(x, avg_holdout, label = "avg holdout error")
        plt.plot([2,4,6,8], std_train, label = "std train error")
        plt.plot([2,4,6,8], std_holdout, label = "std holdout error")
        plt.xticks(np.arange(1, 11, step=1))
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Cross-Entropy Loss: PCA="+str(k)+" learning-rate="+str(learning_rate), fontsize=15)
        plt.legend(loc = 'center left',bbox_to_anchor = (1,0.5))
        
        # calculate average and std of test accuracy
        avg_test = round(np.mean(e_test[k]),2)
        std_test = round(np.std([np.sum(t) for t in e_test[k]]),2)
        print("PCA="+str(k),str(avg_test)+"%", "("+str(std_test)+")")
    plt.show()


# Plot train error, holdout error for 5 runs with different learning rates
# Calculate corresponding test accuracy
def learning_rate(epoch,k_pca,e1,e2,encode,learning_rate):
    e_train = {}
    e_holdout = {}
    e_test = {}
    for i in range(5):
        # choose train set, holdout set, test set 
        train_X,train_t,holdout_X, holdout_t,test_X, test_t = choose_set(e1,e2,encode,i)  

        for rate in learning_rate:
            if rate not in e_train:
                e_train[rate] = []
                e_holdout[rate] = []
                e_test[rate] = []
                
            # transform images to k-dimension
            pca = PCA(k = k_pca)
            pca.fit(train_X)
            train_x = [j[0] for j in [pca.transform(i) for i in train_X]]
            holdout_x = [j[0] for j in [pca.transform(i) for i in holdout_X]]
            test_x = [j[0] for j in [pca.transform(i) for i in test_X]]
         
            e_t, e_h, w_t = run_logistic(epoch,rate,train_x,train_t,holdout_x,holdout_t,test_x,test_t)
            
            # store train error, holdout error for each run
            e_train[rate] += [e_t]
            e_holdout[rate] += [e_h]
            
            # calculate test accuracy
            test_y = logistic_reg(w_t, test_x)
            e_test[rate] += [100*np.mean([(test_y > 0.5) == test_t])]

    for r in learning_rate:
        plt.figure(learning_rate.index(r))
        
        # calculate average of train error, holdout error for 10 epochs
        avg_train = [np.mean([x[i] for x in e_train[r]]) for i in range(10)]
        avg_holdout = [np.mean([x[i] for x in e_holdout[r]]) for i in range(10)]
        
        # calculate std of train error, holdout error for epoch 2,4,6,8
        std_train = [np.std([x[i-1] for x in e_train[r]]) for i in [2,4,6,8]]
        std_holdout = [np.std([x[i-1] for x in e_holdout[r]]) for i in [2,4,6,8]]
    
        # plot graphs
        x = [i for i in range(1,11)]
        plt.plot(x, avg_train, label = "avg train error")
        plt.plot(x, avg_holdout, label = "avg holdout error")
        plt.plot([2,4,6,8], std_train, label = "std train error")
        plt.plot([2,4,6,8], std_holdout, label = "std holdout error")
        plt.xticks(np.arange(1, 11, step=1))
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Cross-Entropy Loss: PCA="+str(k_pca)+" learning-rate="+str(r), fontsize=15)
        plt.legend(loc = 'center left',bbox_to_anchor = (1,0.5))
        
        # calculate average and std of test accuracy
        avg_test = np.mean(e_test[r])
        std_test = round(np.std(e_test[r]),2)
        print("learning rate="+str(r),str(avg_test)+"%", "("+str(std_test)+")")
    plt.show()


# Softmax regression

# Choose images with the 6 emotions(no happy,neutral) from a subject
# build its corresponding expected values using one hot encoding
def subject_emo_softmax(subject,classifier,dataset = dict_label):
    list1 = ['h','m','s','f','a','d']
    x_set = []
    t = []
    for i in subject:
        for l,img in dataset[i]:
            x_set += [img]
            t += [[int(i == list1.index(l[0])) for i in range(classifier)]]

    return (np.array(x_set),np.array(t))


# Choose a pair of subject as test set
# randomly choose a pair of subjects as holdout set from the rest
# then the remaining is train set 
def choose_set_softmax(index, classifier = 6, dataset = dict_label):
    subject = list(dataset.keys())
    
    test = subject[2*index: 2*index+2]
    test_x, test_t = subject_emo_softmax(test,classifier)
    
    rest = subject[:2*index] + subject[2*index+2:]
    random.shuffle(rest)
    
    train = rest[:6]
    train_x, train_t = subject_emo_softmax(train,classifier)
    
    holdout = rest[6:]
    holdout_x, holdout_t = subject_emo_softmax(holdout,classifier)
    
    return train_x,train_t,holdout_x, holdout_t,test_x, test_t


# Compute loss function for softmax regression
def loss_softmax(t_kn, y_kn):
    N = len(t_kn)
    c = len(t_kn[0])
    sumE = 0
    for n in range(N):
        for k in range(c):
            sumE += -(t_kn[n][k])*np.log(y_kn[n][k])
    return sumE/N/c


# Compute predicted values for softmax regression
def softmax_reg(w, x_kn):
    size_N = len(x_kn)
    d = len(w)
    c = len(w[0])
    a_kn = np.zeros([size_N,c])
    y_kn = np.zeros([size_N,c])
    
    for n in range(size_N):
        for k in range(c):
            a_kn[n][k] = np.dot([w[i][k] for i in range(d)],x_kn[n])
        for k in range(c):
            y_kn[n][k] = np.exp(a_kn[n][k])/np.sum([np.exp(a_kn[n][j]) for j in range(c)])
    return y_kn
   

# Compute the train error, holdout error and best weights for each run
def run_softmax(epoch, learning_rate, train_x, train_t, holdout_x, holdout_t, test_x, test_t, mode = "BGD"):

    N = len(train_x)
    c = len(train_t[0])
    d = len(train_x[0])
    
    e_t = []
    e_h = []
    error = 1
    w_t = []

    # weight vector initialization
    w = np.zeros([d,c])
    w_t = np.zeros([d,c])
    
    
    for i in range(epoch):
        
        # batch gradient descent for softmax regression
        if mode == "BGD":
            train_y = softmax_reg(w, train_x)
            e_t += [loss_softmax(train_t, train_y)]
            sumE = np.zeros([d, c])
            for n in range(N):
                m = [[(train_t[n][k] - train_y[n][k])*train_x[n][j] for k in range(c)] for j in range(d)]            
                sumE += m
            w = w + learning_rate/N/c*sumE    
            
        # use stochastic gradient descent for softmax regression
        else:
            P = [i for i in range(N)]
            e_epoch = []
            random.shuffle(P)
            for n in range(N):
                train_y = softmax_reg(w, [train_x[P[n]]])
                
                e_epoch += [loss_softmax([train_t[P[n]]], train_y)]
                m = [[(train_t[P[n]][k] - train_y[0][k])*train_x[P[n]][j] for k in range(c)] for j in range(d)]
                w = w + np.array(m)*learning_rate/c
            e_t += [np.mean(e_epoch)]
        
        # record holdout error and best weights
        holdout_y = softmax_reg(w, holdout_x)
        e_h += [loss_softmax(holdout_t, holdout_y)] 
        
        if e_h[-1] <= error:
            error = e_h[-1]
            w_t = w
    
    return e_t, e_h, w_t


# Plot train error, holdout error for 5 runs with different PCAs
# Calculate corresponding confusion matrix
def training_proc_softmax(epoch, kd,learning_rate):
    e_train = {}
    e_holdout = {}
    e_test = {}
    w = {}
    confusion = {}
    
    for i in range(5):
        # choose train set, holdout set, test set 
        train_X,train_t,holdout_X, holdout_t,test_X, test_t = choose_set_softmax(i)  
       
        for k_pca in kd:
            if k_pca not in e_train:
                e_train[k_pca] = []
                e_holdout[k_pca] = []
                e_test[k_pca] = []
                w[k_pca] = []
                confusion[k_pca] = np.zeros([6,6])
                
            # transform images to k-dimension
            pca = PCA(k = k_pca)
            pca.fit(train_X)
            train_x = [j[0] for j in [pca.transform(i) for i in train_X]]
            holdout_x = [j[0] for j in [pca.transform(i) for i in holdout_X]]
            test_x = [j[0] for j in [pca.transform(i) for i in test_X]]

            e_t, e_h, w_t = run_softmax(epoch,learning_rate,train_x,train_t,holdout_x,holdout_t,test_x,test_t)
    
            # store train error, holdout error for each run
            e_train[k_pca] += [e_t]
            e_holdout[k_pca] += [e_h]
            
            # calculate test accuracy
            test_y = softmax_reg(w_t, test_x)
            for t in range(len(test_y)):
                maxt = max(test_y[t])
                test = list(test_y[t]).index(maxt)
                correct = list(test_t[t]).index(1)
                confusion[k_pca][correct][test] += 1
            
            w[k_pca] += [w_t]
    
    # calculate confusion matrix
    for k_pca in kd:
        confusion[k_pca] = np.array([c/sum(c)*100 for c in confusion[k_pca]])
        print("confusion matrix: ")
        print(confusion[k_pca])
    
    for k in kd:
        plt.figure(kd.index(k))
        
        # calculate average of train error, holdout error for 20 epochs
        avg_train = [np.mean([x[i-1] for x in e_train[k]]) for i in range(1,21)]
        avg_holdout = [np.mean([x[i-1] for x in e_holdout[k]]) for i in range(1,21)]
        
        # calculate std of train error, holdout error for epoch 5,10,15,20
        std_train = [np.std([x[i-1] for x in e_train[k]]) for i in [5,10,15,20]]
        std_holdout = [np.std([x[i-1] for x in e_holdout[k]]) for i in [5,10,15,20]]
        
        # plot graphs
        x = [i for i in range(1,21)]
        plt.plot(x, avg_train, label = "train error")
        plt.plot(x, avg_holdout, label = "holdout error")
        plt.plot([5,10,15,20], std_train, label = "std train error")
        plt.plot([5,10,15,20], std_holdout, label = "std holdout error")
        plt.xticks(np.arange(1, 21, step=1))
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Cross-Entropy Loss: PCA="+str(k)+" learning-rate="+str(learning_rate), fontsize=15)
        plt.legend(loc = 'center left',bbox_to_anchor = (1,0.5))
    plt.show()


# Plot train error curves for batch gradient descent and for stochastic gradient descent
def BGD_SGD(epoch, kd,learning_rate):
    e_train = {}
    e_holdout = {}
    e_test = {}
    
    for i in range(5):
        # choose train set, holdout set, test set
        train_X,train_t,holdout_X, holdout_t,test_X, test_t = choose_set_softmax(i)  
       
        # choose mode batch gradient descent or stochastic gradient descent
        for mode in ["BGD","SGD"]:
            if mode not in e_train:
                e_train[mode] = []
                e_holdout[mode] = []
                e_test[mode] = []
                
            # transform images to k-dimension
            pca = PCA(k = kd)
            pca.fit(train_X)
            train_x = [j[0] for j in [pca.transform(i) for i in train_X]]
            holdout_x = [j[0] for j in [pca.transform(i) for i in holdout_X]]
            test_x = [j[0] for j in [pca.transform(i) for i in test_X]]

            e_t, e_h, w_t = run_softmax(epoch, learning_rate,train_x,train_t,holdout_x,holdout_t,test_x,test_t, mode)
            
            # store train error, holdout error for each run
            e_train[mode] += [e_t]
            e_holdout[mode] += [e_h]


    for mode in ["BGD","SGD"]:
        # calculate average of train error for 20 epochs
        avg_train = [np.mean([x[i] for x in e_train[mode]]) for i in range(epoch)]

        # plot graphs
        x = [i for i in range(1,epoch+1)]
        plt.plot(x, avg_train, label = mode,markersize=5)
        plt.xticks(np.arange(1, 21, step=1))
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Cross-Entropy Loss: PCA="+str(kd)+" learning-rate="+str(learning_rate), fontsize=15)
        plt.legend(loc = 'upper right')
    plt.show()


# linear scale a vector to make the minimum value is 0, maximum value is 255
def linearscale(vec):
    minV = np.min(vec)
    maxV = np.max(vec)
    a = 255/(maxV-minV)
    b = -minV*a
    return [a*v+b for v in vec]


# visualization of weight
def visualize(epoch,kd,learning_rate):
    w = []
    list1 = ['happy with teeth','maudlin','surprise','fear','angry','disgust']
    inverse = {}
    
    for i in range(5):
        # choose train set, holdout set, test set
        train_X,train_t,holdout_X, holdout_t,test_X, test_t = choose_set_softmax(i)      
      
        # transform images to k-dimension
        pca = PCA(k = kd)
        pca.fit(train_X)
        train_x = [j[0] for j in [pca.transform(i) for i in train_X]]
        holdout_x = [j[0] for j in [pca.transform(i) for i in holdout_X]]
        test_x = [j[0] for j in [pca.transform(i) for i in test_X]]

        e_t, e_h, w_t = run_softmax(epoch,learning_rate,train_x,train_t,holdout_x,holdout_t,test_x,test_t)
        
        # for each emotion, transform the corresponding weight to the original image representation
        for l in list1:
            if (l not in inverse):
                inverse[l] = []
            k = list1.index(l)
            w_k = [cate[k] for cate in w_t]
            inverse[l] += [pca.inverse_transform([[w_k]])]
    
    for l in list1:
        inverse[l] = np.mean(inverse[l], axis = 0)
        # linear scale
        scale = linearscale(inverse[l])
        plt.title(l)
        plt.imshow(np.array(scale), cmap='gray')
        plt.show()


# Extra Credit

# Choose images with the specified emotions from all the 10 subjects
# build its corresponding expected values using one hot encoding
def subject_emo_ec(emotion,classifier,dataset = dict_label_ec):
    list1 = ['018', '027', '036', '037', '041', '043', '044', '048ng', '049', '050']

    x_set = []
    t = []
    for i in emotion:
        for l,img in dataset[i]:
            x_set += [img]
            t += [[int(j == list1.index(l)) for j in range(classifier)]]
    return (np.array(x_set),np.array(t))


# Choose one emotion of all 10 subjects as test set
# randomly choose anothor emotion of all 10 subjects as holdout set from the rest
# then the remaining is train set 
def choose_set_ec(index, classifier = 6):
    emotions = ['h','m','s','f','a','d']
    
    test = emotions[index : index+1]
    test_x, test_t = subject_emo_ec(test,classifier)
    
    rest = emotions[:index] + emotions[index+1:]
    random.shuffle(rest)
    
    holdout = rest[:1]
    holdout_x, holdout_t = subject_emo_ec(holdout,classifier)

    train = rest[1:]
    train_x, train_t = subject_emo_ec(train,classifier)

    return train_x,train_t,holdout_x, holdout_t,test_x, test_t


# Plot train error, holdout error for 6 runs with different PCAs
def training_proc_ec(epoch, kd,learning_rate):
    e_train = {}
    e_holdout = {}
    e_test = {}
    w = {}
    
    for i in range(6):
        # choose train set, holdout set, test set
        train_X,train_t,holdout_X, holdout_t,test_X, test_t = choose_set_ec(i,10)  
       
        for k_pca in kd:
            if k_pca not in e_train:
                e_train[k_pca] = []
                e_holdout[k_pca] = []
                e_test[k_pca] = []
                w[k_pca] = []
            
            # transform images to k-dimension
            pca = PCA(k = k_pca)
            pca.fit(train_X)
            train_x = [j[0] for j in [pca.transform(i) for i in train_X]]
            holdout_x = [j[0] for j in [pca.transform(i) for i in holdout_X]]
            test_x = [j[0] for j in [pca.transform(i) for i in test_X]]

            e_t, e_h, w_t = run_softmax(epoch,learning_rate,train_x,train_t,holdout_x,holdout_t,test_x,test_t)
    
            # store train error, holdout error for each run
            e_train[k_pca] += [e_t]
            e_holdout[k_pca] += [e_h]
            
            test_y = softmax_reg(w_t, test_x)            
            w[k_pca] += [w_t]
 
    for k in kd:
        plt.figure(kd.index(k))
        
        # calculate average of train error for 20 epochs
        avg_train = [np.mean([x[i-1] for x in e_train[k]]) for i in range(1,21)]
        avg_holdout = [np.mean([x[i-1] for x in e_holdout[k]]) for i in range(1,21)]
        
        # calculate std of train error for epoch 5,10,15,20
        std_train = [np.std([x[i-1] for x in e_train[k]]) for i in [5,10,15,20]]
        std_holdout = [np.std([x[i-1] for x in e_holdout[k]]) for i in [5,10,15,20]]
        
        # plot graphs
        x = [i for i in range(1,21)]
        plt.plot(x, avg_train, label = "train error")
        plt.plot(x, avg_holdout, label = "holdout error")
        plt.plot([5,10,15,20], std_train, label = "std train error")
        plt.plot([5,10,15,20], std_holdout, label = "std holdout error")
        plt.xticks(np.arange(1, 21, step=1))
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Cross-Entropy Loss: PCA="+str(k)+" learning-rate="+str(learning_rate), fontsize=15)
        plt.legend(loc = 'center left',bbox_to_anchor = (1,0.5))
    plt.show()
    
    
    
# visualization of weight
def visualize_ec(epoch, kd, learning_rate):
    w = []
    list1 = ['018', '027', '036', '037', '041', '043', '044', '048ng', '049', '050']

    inverse = {}
    
    for i in range(6):
        # choose train set, holdout set, test set
        train_X,train_t,holdout_X, holdout_t,test_X, test_t = choose_set_ec(i,10)      
           
        # transform images to k-dimension
        pca = PCA(k = kd)
        pca.fit(train_X)
        train_x = [j[0] for j in [pca.transform(i) for i in train_X]]
        holdout_x = [j[0] for j in [pca.transform(i) for i in holdout_X]]
        test_x = [j[0] for j in [pca.transform(i) for i in test_X]]

        e_t, e_h, w_t = run_softmax(epoch,learning_rate,train_x,train_t,holdout_x,holdout_t,test_x,test_t)

        # for each subject, transform the corresponding weight to the original image representation
        for l in list1:
            if (l not in inverse):
                inverse[l] = []
            k = list1.index(l)
            w_k = [cate[k] for cate in w_t]
            inverse[l] += [pca.inverse_transform([[w_k]])]
        
    for l in list1:
        inverse[l] = np.mean(inverse[l], axis = 0)
        # linear scale
        scale = linearscale(inverse[l])
        plt.title(l)
        plt.imshow(np.array(scale), cmap='gray')
        plt.show()



def main():
    print("Part1")
    print("*****1.(b)*****")
    # display 6 different emotions from one subject
    for i in dict_label['018']:
        plt.imshow(i[1], cmap='gray')
        plt.show()
    
    print("*****1.(c)*****")
    # display first 6 eigenvectors
    # k is the number of principal components 
    pca = PCA(k=50)
    # choose training images
    pca.fit(np.array(images[:]))
    pca.display('./1c_pca_display.png', 6)
    plt.show()

    print("Part2")
    print("*****2.(b)*****")
    # experiment different PCA
    training_proc(10, [1,2,4,8],'h','m',1,2)
    # experiment different learning rate
    learning_rate(10, 8,'h','m',1,[0.01,3,10])
    
    print("*****2.(c)*****")
    # afraid(fear) vs. surprise
    learning_rate(10, 8,'f','s',1,[3]) 
    
    print("Part3")
    print("*****3.(a)*****")
    # experiment different PCA and learning rate
    training_proc_softmax(20, [10,20,30], 3)
    training_proc_softmax(20, [10,20,30], 5)
    
    print("*****3.(b)*****")
    # experiment batch gradient descent, stochastic gradient descent
    BGD_SGD(20, 30, 5)
    
    print("*****3.(c)*****")
    # visualize weight
    visualize(20,30,5)
    
    print("*****3.(d)*****")
    # experiment different learning rate
    training_proc_ec(20, [30], 5)
    training_proc_ec(20, [30], 10)
    training_proc_ec(20, [30], 20)
    # visualize weight
    visualize_ec(20,30,20)


if __name__ == '__main__':
	main()




