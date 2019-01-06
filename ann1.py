# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 01:34:05 2017

@author: Sahithi
"""
import numpy as np
from nltk import word_tokenize
import re
import time
from nltk.stem import PorterStemmer
# 3 classes of training data
training_data = []
training_data.append({"class":"greeting", "sentence":"how are you?"})
training_data.append({"class":"greeting", "sentence":"how is your day?"})
training_data.append({"class":"greeting", "sentence":"good day"})
training_data.append({"class":"greeting", "sentence":"how is it going today?"})

training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"see you later"})
training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"talk to you soon"})

training_data.append({"class":"sandwich", "sentence":"make me a sandwich"})
training_data.append({"class":"sandwich", "sentence":"can you make a sandwich?"})
training_data.append({"class":"sandwich", "sentence":"having a sandwich today?"})
training_data.append({"class":"sandwich", "sentence":"what's for lunch?"})
print ("%s sentences in training data" % len(training_data))

words=[]
classes=[]
docs=[]
for cols in training_data:
    if cols["class"] not in classes:
        classes.append(cols["class"])
        
for cols in training_data:
    tokens=word_tokenize(cols["sentence"])
    #Apostrophe lookup
    Appost_dict={"'s":"is","'re":"are","'ve":"have","n't":"not","d":"had","'ll":"will","'m":"am"}
    reformed=[Appost_dict[word] if word in Appost_dict else word for word in tokens]
    processed=[word.lower() for word in reformed if word != '?']
    words.extend(processed)
    docs.append((processed,cols["class"]))
    

    
##Removing duplicates
words=list(set(words))
###Creating bag of words
training=[]
output=[]
output_class=[0 for i in classes]


for doc in docs:
    bag=[]
    input_words=doc[0]
    print(input_words)
    i_words=[word for word in input_words]
    for word in words:
        if word in i_words:
            bag.append(1)
        else:
            bag.append(0)
    training.append(bag)
    output_class[classes.index(doc[1])]=1
    output.append(output_class)


def cleanup(sentence):
    #Tokenize++++++++
    token=word_tokenize(sentence)
    #Apostrophe lookup
    Appost_dict={"'s":"is","'re":"are","'ve":"have","n't":"not","d":"had","'ll":"will","'m":"am"}
    reformed=[Appost_dict[word] if word in Appost_dict else word for word in token]
    #lower case
    processed=[word.lower() for word in reformed if word != '?']
    return processed

def bagofwords(sentence_words,words,showdetails="False"):
    bags=[0 for i in range(len(words))]
    for i in sentence_words:
        for j,k in enumerate(words):
            if(i==k):
                bags[j]=1
                if showdetails:
                    print(" Word %s found in bag\n" %i)
    return bags
def sigmoid(x):
    op=1/(1+np.exp(-x)).
    return op
def derivative(x):
    derv=x*(1-x)
    return derv
def classify(sentence,w0,w1):
    proc_sent=cleanup(sentence)
    bag=bagofwords(proc_sent,words,"True")
    x=bag #Input is our bag of words
    L0=np.array(x) # layer 0 
    L1=sigmoid(np.dot(L0,w0)) #Layer 1
    L2=sigmoid(np.dot(L1,w1)) # Layer 2
    return L2
    
    
##Method for training Neural Network
def train(X,y,hidden_neurons=10,alpha=0.1,epochs=10000):
    (m,n)=X.shape
    print("Order of Input matrix is: ")
    print(X.shape)
    (a,b)=y.shape
    print("Order of output matrix is:")
    print(y.shape)
    np.random.seed(1)
    ##Intialize weights
    w0=2*np.random.random((n,hidden_neurons))-1 # For Layer 0
    w1=2*np.random.random((hidden_neurons,b))-1 # For Layer 1
    ##Layers
    for i in range(epochs):
        L_0=X
        L_1=sigmoid(np.dot(L_0,w0))
    
        L_2=sigmoid(np.dot(L_1,w1))
        ##Error at Layer 2
        l2_error=y-L_2 # how much are we missing target
        l2_delta=l2_error*derivative(L_2)
        l1_error=np.dot(l2_delta,w1.T)
        l1_delta=l1_error*derivative(L_1)
        ##Weight update
        w0_upd=np.dot(L_0.T,l1_delta)
        w1_upd=np.dot(L_1.T,l2_delta)
        ##Updating weights
        w0=w0+(alpha*w0_upd)
        w1=w1+(alpha*w1_upd)
    return w0,w1
##Inputs for training Neural Network
X=np.array(training)
y=np.array(output)
start_time=time.time()
w0,w1=train(X,y,hidden_neurons=20,alpha=0.1,epochs=1000000)
print("Value of w0 is:")
print(w0)
print("Value of w1 is:")
print(w1) 

result=classify("talk to you tomorrow",w0,w1)
results=[[i,r] for i,r in enumerate(result)]
results.sort(key=lambda x:x[1],reverse=True)
#class_ret=[[classes[r[0]],r[1]] for r in results]
class_ret=[[classes[results[0][0]]]]
print(class_ret)
    




    
    



    


