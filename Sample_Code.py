
import numpy as np
from nltk import word_tokenize
import re
import time
import csv
import matplotlib.pyplot as plt #used for plotting graphs
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot #used for plotting graphs
from plotly.graph_objs import *   #used for plotting graphs
import numpy as np #used for columnstack
import re #used for pre-processing 
from html.parser import HTMLParser # used to remove the html tags
from nltk.corpus import stopwords # used to remove Stopwords
from nltk.tokenize import TweetTokenizer #used for tokenization
from nltk.tokenize import TreebankWordTokenizer #used for tokenization
from sklearn.metrics import * # used to calculate performance metrics
from sklearn.model_selection import train_test_split

##Function to eread input file
def read_data(filename):
    with open(filename,encoding='utf8') as tsv:
        trainTweet = [line.strip().split('\t') for line in tsv]
        return trainTweet

##Load input file
data= read_data('Sample_data.txt')

##Function for preprocessing input statements
def preprocessing(original_sentence):
    ##1.TreebankTokenizer
    res1=TreebankWordTokenizer().tokenize(original_sentence)
    print("After Tokenizing------------")
    print(res1)
    ##2.Contractions Removal  
    Appost_dict={"'s":"is","'re":"are","'ve":"have","n't":"not","d":"had","'ll":"will","'m":"am",}
    transformed=[Appost_dict[word].lower() if word in Appost_dict else word for word in res1]
    print("After handling contractions-------")
    print(transformed)
    ##3.Special Characters Removal 
    res2=" ".join(transformed)
    res3=re.sub(r"[!@#$%^&*()_+-=:;?/~`'’]",' ',res2)
    print("After removing unwanted characters-----------")
    print(res3)             
    ##4.Tweet tokenizer
    tkznr=TweetTokenizer(reduce_len=False,strip_handles=True,preserve_case=False)
    res4=tkznr.tokenize(res3)
    print("After tweet tokenizing---")
    print(res4)
    ##5.Stopwords Removal
    processed_sentence=[word for word in res4 if word not in stopwords.words('english')] 
    print("After removing stopwords-----")
    print(processed_sentence)
    return processed_sentence


words=[] ##To store words from all excerpts
authors=[] # To store distint author names
excerpts=[] #To store excerpts from training data

#Splitting Training data wth 70% and test data with 30%
train_data,test_data=train_test_split(data,test_size=0.3)

##For Training Data
for cols in train_data:
    if cols[2] not in authors:
        authors.append(cols[2]) #Store distinct authors
    proc_excerpts=preprocessing(cols[1]) # Process input sentences
    words.extend(proc_excerpts) # Store all words 
    excerpts.append((proc_excerpts,cols[2])) #Store the tokens of excerpts with associated author

##For Test Data
test_excerpt=[] # Store test excerpts
actual_author=[] # To Store actual test author data
for cols in test_data:
    proc=preprocessing(cols[1]) #Preprocessing test excerpts
    test_excerpt.append(proc) 
    actual_author.append(cols[2]) #Storing author of the excerpt

##Removing duplicate words
words=list(set(words))
training_bag=[] #Holds Unigrams for training data
author=[] #holds author of each sentence in an array

#Building Bag of words for Training data in terms of 0s and 1s
for cols in excerpts:
    output_author=[0 for i in authors] #Array indicating author of each input sentence
    bag=[]
    input_words=cols[0]
    i_words=[word for word in input_words]
    for word in words:
        # If words from Training Data is present in corpus
        if word in i_words:
            bag.append(1) #Indicating the presence of word with 1
        else:
            bag.append(0) #Indicating the absence of word with 0 
    training_bag.append(bag)
    output_author[authors.index(cols[1])]=1
    author.append(output_author)

#Funtion to create bag of words for test data
def bagofwords(sentence_words,words,showdetails="False"):
    bags=[0 for i in range(len(words))]
    for i in sentence_words:
        for j,k in enumerate(words):
            if(i==k):
                bags[j]=1
                #if showdetails:
                 #   print(" Word %s found in bag\n" %i)
    return bags
#Funtion to create non-linear exponential function to normalise values
def sigmoid(x):
    op=1/(1+np.exp(-x))
    return op

#Function to calculate error rate
def derivative(x):
    derv=x*(1-x)
    return derv

#Function to classify sentiment of test data
def classification(test_excerpts,w0,w1):
    ret=[]
    class_return=[]
    for exc in test_excerpts:
        bag=bagofwords(exc,words,"True") #step to create bag of words for test data based on training data in terms of 0s and 1s
        x=bag #Input is our bag of words
        L0=np.array(x) # layer 0 of Neural Network
        L1=sigmoid(np.dot(L0,w0)) #Layer 1 of Neural Network
        L2=sigmoid(np.dot(L1,w1)) # Layer 2 of Neural Network with probabilities of classes the input belongs to
        probability=[[i,r] for i,r in enumerate(L2)]
        probability.sort(key=lambda x:x[1],reverse=True)# Sorting the class probabilities to pick up the highest probability class
        ret=authors[probability[0][0]]
        class_return.append(ret)# return class with highest probability
    return class_return

#Function to train Neural Network
def train(X,y,hidden_neurons=10,alpha=0.1,iterations=10000):
    (m,n)=X.shape # Bag of words of training Data
    (a,b)=y.shape # Sentiment associated with training data
    #Initialising weights with random numbers
    np.random.seed(1)
    #Intialize weights
    w0=2*np.random.random((n,hidden_neurons))-1 # For Layer 0
    w1=2*np.random.random((hidden_neurons,b))-1 # For Layer 1
    #Building Layers of Neural Network
    for i in range(iterations):
        L_0=X # Bag of words as input
        L_1=sigmoid(np.dot(L_0,w0)) # Function to form layer 1
        L_2=sigmoid(np.dot(L_1,w1)) # Function to form Layer 2
        #To find Error at Layer 2
        L2_error=y-L_2 # determining how much our prediction is differing from actual value at Layer 2
        L2_delta=L2_error*derivative(L_2) #Direction in which prediction is missing
        L1_error=np.dot(L2_delta,w1.T) #Determining how much our prediction is differing from actual value due to Layer 1
        L1_delta=L1_error*derivative(L_1) #Direction in which prediction is missing
        #Updating weights as per error rate
        w0_upd=np.dot(L_0.T,L1_delta)
        w1_upd=np.dot(L_1.T,L2_delta)
        w0=w0+(alpha*w0_upd)
        w1=w1+(alpha*w1_upd)
    return w0,w1  

##Inputs for training Neural Network
X=np.array(training_bag)
y=np.array(author)
# To obtain Weights by training Neural Network
w0,w1=train(X,y,hidden_neurons=30,alpha=0.01,iterations=500)
print("Value of w0 is:")
print(w0)
print("Value of w1 is:")
print(w1) 



#Predicting sentiment of Test Data
predicted_author=[]
predicted_author=classification(test_excerpt,w0,w1) # Classifying Input Test data
conf_matrix=confusion_matrix(actual_author,predicted_author)
print("**********************Confusion Matrix*****************************")
print(conf_matrix)
accuracy=accuracy_score(actual_author,predicted_author)
print("Accuracy of classifier is:")
print(accuracy)
print("**********************Classification report************************")
print(classification_report(actual_author,predicted_author))


    



    
    



        
     

