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
ps=PorterStemmer()
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
train=[]
output=[]
output_class=[0 for i in classes]


for doc in docs:
    bag=[]
    input_words=doc[0]
    print(input_words)
    i_words=[word for word in input_words]
    print(i_words)
    for word in words:
        if word in i_words:
            bag.append(1)
        else:
            bag.append(0)
    train.append(bag)
    output_class[classes.index(doc[1])]=1
    output.append(output_class)



X=np.array(train)
(m,n)=X.shape
print(m)