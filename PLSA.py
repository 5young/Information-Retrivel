import glob
import pandas as pd
import numpy as np
import math
import os
import random
import cProfile
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from scipy.special import logsumexp


def buildDictionary():
    #file = open('Collection.txt')
    file = open(r'C:\Users\young\Desktop\IR\Collection.txt')
    file = file.read()
    wordList = file.split("\n")
    print(len(wordList))
    wordDictionary = []
    for i in range(len(wordList)):
        string = wordList[i].split(" ")
        for j in range(len(string)):
            if (string[j] != ''):
                wordDictionary.append(string[j])
    wordDictionary = np.array(wordDictionary)
    wordDictionary = np.unique(wordDictionary)  
    return wordDictionary, wordList

def logsoftmax(x):

    return x - np.logaddexp.reduce(x)

def create_tf(list1, list2):
    tf_array = np.zeros((len(list1), len(list2)), float)
    for i in range(len(list1)):
        print ("create tf table", format((i/len(list1)*100), '.2f'), "%")
        for j in range(len(list2)):
            if list2[j].count(list1[i]) == 0:
                tf_array[i, j] = 0.00000000001
            else:
                tf_array[i, j] = list2[j].count(list1[i])
    np.save("tf_array", tf_array)
    return tf_array

def e_step(i ,j):
    if j == -1: # return t x d matrix
        #e1_logsumexp_array = np.log(e1_duplicate_vector(w_t_array[i, :])) + np.log(t_d_array[:, :])
        #print(e1_logsumexp_array)
        e1_logsumexp_array = w_t_array[i, :][:, None] + t_d_array[:, :]
        prob_t_wd1 = w_t_array[i, :][:, None] + t_d_array[:, :] - logsumexp(e1_logsumexp_array)
        return prob_t_wd1

    if i == -1: # return w x t matrix
        e2_logsumexp_array = w_t_array[:, :] + t_d_array[:, j][None, :]
        prob_t_wd2 = w_t_array[:, :] + t_d_array[:, j][None, :] - logsumexp(e2_logsumexp_array)
        print("#########", prob_t_wd2.shape)
        return prob_t_wd2
    
def m_step1(len1, len2):
    for i in range(len1):
        m_step1_array[i, :, :] = tf_array[i, :][:, None] + e_step(i, -1).T
    result1 = logsumexp(m_step1_array[i, :, 0]) - logsumexp(m_step1_array) 
    result2 = logsumexp(m_step1_array[i, :, 1]) - logsumexp(m_step1_array)
    result = np.array([result1, result2])
    return result

def m_step2(len1, j):
    result1 = 0
    result2 = 0
    count = np.zeros((wordDictionary_len, 1), float)
    m_step2_array = e_step(-1, j) + tf_array[:, j][:, None]
    count = tf_array[:, j].sum()
    result1 = logsumexp(m_step2_array[:, 0]) - np.log(count)
    result2 = logsumexp(m_step2_array[:, 1]) - np.log(count)
    result = np.array([result1, result2])
    # result = np.zeros((k, 1), float)
    # result[0, 0] = result1
    # result[1, 0] = result2
    return result



wordDictionary, wordList = buildDictionary()
wordDictionary_len = len(wordDictionary) #total word len
wordList_len = len(wordList)
print("wordDictionary_len = ", wordDictionary_len)
print("wordList_len = ", wordList_len)
k = 2
iterations = 2 #value of end condition
iteration = 0 

# initial array
w_t_array = np.zeros((wordDictionary_len, k), float) #w_t table
t_d_array = np.zeros((k, wordList_len), float) #t_d table
new_w_t_array = np.zeros((wordDictionary_len, k), float) #new w_t table
new_t_d_array = np.zeros((k, wordList_len), float) #new t_d table
m_step1_array = np.zeros((wordDictionary_len, wordList_len, 2), float) #w_d table
m_step2_array = np.zeros((wordDictionary_len, k), float)
tf_array = create_tf(wordDictionary, wordList)
#tf_array = np.load("tf_array.npy")
tf_array = np.log(tf_array)
prob_t_wd1 = np.zeros((2, wordList_len), float)
prob_t_wd2 = np.zeros((wordDictionary_len, 2), float)

while (iteration < iterations):
    # random first value and softmax in w_t table
    for i in range(k):
        print ("w_t table", format((i/k*100), '.2f'), "%")
        for j in range(wordDictionary_len):
            if iteration == 0:
                w_t_array[j, i] = np.log(random.uniform(0, 1))
        w_t_array[:, i] = logsoftmax(w_t_array[:, i])

    for i in range(wordList_len):
        print ("t_d table", format((i/wordList_len*100), '.2f'), "%")
        for j in range(k):
            if iteration == 0:
                t_d_array[j, i] = np.log(random.uniform(0, 1))
        t_d_array[:, i] = logsoftmax(t_d_array[:, i])

    for i in range(wordDictionary_len):
        print ("em_top", format((i/wordDictionary_len*100), '.2f'), "%")
        new_w_t_array[i, :] = m_step1(wordDictionary_len, wordList_len)

    for j in range(wordList_len):
        print ("em_bottom", format((j/wordList_len*100), '.2f'), "%")
        new_t_d_array[:, j] = m_step2(wordDictionary_len, j)

    w_t_array = new_w_t_array
    t_d_array = new_t_d_array

    iteration = iteration + 1