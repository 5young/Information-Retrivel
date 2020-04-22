import glob
import pandas as pd
import numpy as np
import math
import os
from sklearn.feature_extraction.text import CountVectorizer

def openDocFile(path_list):
    print("open the document file")
    wordList = []
    for i in range(len(path_list)):
        f = open(path_list[i], 'r')
        s = f.read()
        str_list = s.split("\n")
        del str_list[0]
        del str_list[0]
        del str_list[0]
        str1 = ''.join(str_list) #list轉string
        str1 = str1.replace("-1", "")
        str1 = str1.strip()
        list1 = str1.split(" ")
    
        for j in range(0, len(list1)):
            wordList.append(list1[j])
    f.close()
    print("document file already open")
    return wordList

def openQueryFile(path_list):
    print("open the query file")
    wordList = []
    for i in range(len(path_list)):
        f = open(path_list[i], 'r')
        s = f.read()
        str_list = s.split("\n")
        str1 = ''.join(str_list) #list轉string
        str1 = str1.replace("-1", "")
        str1 = str1.strip()
        list1 = str1.split(" ")
    
        for j in range(0, len(list1)):
            wordList.append(list1[j])
    print("query file already open")
    return wordList

def countDict(wordList):
    print("counting")
    word_count_dict = {}
    for i in range(len(wordList)):
        word_count_dict[wordList[i]] = 0
    for i in range(len(wordList)):
        word_count_dict[wordList[i]] += 1

    #print (format(i/(len(wordList) * 100, '.2f'), "%"))
    print(word_count_dict)
    print("dict size = " , len(word_count_dict))
    
    #os.system("pause")
    return word_count_dict

def buildDictionary(listX):
    #print("正在創建字典")
    word_keyList = [] #wword_count_dict裡的key
    word_valueList = [] #wword_count_dict裡的value
    for key, value in listX.items():
        word_keyList.append(key)
        word_valueList.append(value)
        print("building dictionary")

    wordIndex = list(range(len(word_keyList)))
    print(type(dict(zip(wordIndex, word_keyList))))
    return dict(zip(wordIndex, word_keyList)), word_keyList

def inverseDocFreq(word_dict, word): #這裡傳入的wordlist是尚未統計次數的
    #dict1 = countDict(wordList)
    #print(type(word_dict[word]))
    return math.log(2265/word_dict[word])

def createDocArr(doc_list, wordKeyList):
    doc_arr = np.zeros((len(wordKeyList), 2265), float) #創建doc表格
    for i in range(len(doc_list)):
        print ("bule doc table", format((i/(len(doc_list))*100), '.2f'), "%")
        f = open(doc_list[i], 'r')
        s = f.read()
        str_list = s.split("\n")
        del str_list[0]
        del str_list[0]
        del str_list[0]
        str1 = ''.join(str_list) #list轉string
        str1 = str1.replace("-1", "")
        str1 = str1.strip()
        list1 = str1.split(" ")

        for j in range(len(wordKeyList)):
            #doc_arr[j, i] = list1.count(wordKeyList[j])
            doc_arr[j, i] = list1.count(wordKeyList[j])
        f.close()

    for i in range(len(doc_list)): #將tf乘上idf
        print ("doc tf * idf", format((i/(len(doc_list))*100), '.2f'), "%")
        for j in range(len(wordKeyList)):

            numOfidf = np.count_nonzero(doc_arr[j, :])
            #print(numOfidf)
            if numOfidf == 0: #防止IDF為0
                doc_arr[j, i] = 0
            else: 
                doc_arr[j, i] = doc_arr[j, i] * math.log(2265/numOfidf)
            #print("idf weight = ", math.log(2265/numOfidf))

    return doc_arr

def createQueryArr(query_list, wordKeyList):
    query_arr = np.zeros((len(wordKeyList), 16), float) #創建query表格
    for i in range(len(query_list)):
        print ("build query table", format((i/(len(query_list))*100), '.2f'), "%")
        f = open(query_list[i], 'r')
        s = f.read()
        str_list = s.split("\n") 
        str1 = ''.join(str_list) #list轉string
        str1 = str1.replace("-1", "")
        str1 = str1.strip()
        list1 = str1.split(" ")

        for j in range(len(wordKeyList)):
            query_arr[j, i] = list1.count(wordKeyList[j])

        f.close()

    for i in range(len(query_list)): #將tf乘上idf
        print ("query tf * idf", format((i/(len(query_list))*100), '.2f'), "%")
        for j in range(len(wordKeyList)):

            numOfidf = np.count_nonzero(query_arr[j, :])
            if numOfidf == 0: #防止IDF為0
                query_arr[j, i] = 0
            else:
                query_arr[j, i] = query_arr[j, i] * math.log(2265/numOfidf)

    return query_arr

def createVsmArr(query_list, doc_list):
    for i in range(0, len(query_list)):
        print ("build vsm file", format((i/(len(query_list))*100), '.2f'), "%")
        queryVector = query_arr[:, i]
        for j in range(0, len(doc_list)):
            a = 0
            b = 0
            c = 0
            docVector = doc_arr[:, j]
            
            for k in range(len(queryVector)):
                a += queryVector[k] * queryVector[k]
                b += docVector[k] * docVector[k]
                c += queryVector[k] * docVector[k]

            vsm_arr[i, j] = c / ((a ** 0.5) * (b ** 0.5))

    return vsm_arr


def fileWrite(query_list, doc_list):
    result = open("result.txt", 'w')
    result.write("Query,RetrievedDocuments\n")
    for i in range(len(query_list)):
        doc_name_list = []

        for j in range(len(doc_list)):
            doc_name_list.append(doc_list[j])

        print("doc_name_list = ", doc_name_list)
        vsm_list = vsm_arr[i, :]
        vsm_dict = dict(zip(vsm_list, doc_name_list)) #sorted預設用前面的作sort
        vsm_list = sorted(vsm_list, reverse = True)
    
        result.write(query_list[i].replace("C:\\Users\\young\\Desktop\\IR\Query\\", "") + ",")
        for k in range(len(vsm_list)):
            resultStr = vsm_dict[vsm_list[k]]
            resultStr = resultStr.replace("C:\\Users\\young\\Desktop\\IR\\Document\\", "")
            result.write(resultStr + " ")

        result.write("\n")

    result.close()


############################
############main############
doc_list  = glob.glob(r'C:\Users\young\Desktop\IR\Document\*') #文檔檔名list          
query_list = glob.glob(r'C:\Users\young\Desktop\IR\Query\*') #query檔檔名list
#print(doc_list)
wordList1 = openDocFile(doc_list) #文檔全部的字
print(len(wordList1))
wordList2 = openQueryFile(query_list) #query檔全部的字
print(len(wordList2))
wordList = wordList1 + wordList2 #用文檔與query檔裡全部的字做字典 以及計算次數
#print(openDocFile(doc_list))
word_dict = countDict(wordList) #word在文檔中的總出現次數
#print(word_dict)
dictionary, wordKeyList = buildDictionary(word_dict) #得到字典及文檔中的所有字
#doc_arr = np.zeros((len(wordKeyList), 2265), float) #創建doc表格 改成在function裡創
doc_arr = createDocArr(doc_list, wordKeyList) #填入doc表格
print("doc table complete")
print("doc_arr = ", doc_arr)
print("----------------------------------------------------------")
#query_arr = np.zeros((len(wordKeyList), 16), float) #創建query表格 改成在function裡創
query_arr = createQueryArr(query_list, wordKeyList) #填入query表格
print("query table complete")
print("query_arr = ", query_arr)
print("----------------------------------------------------------")

vsm_arr = np.zeros((len(query_list), len(doc_list)), float) #創建vsm表格
vsm_arr = createVsmArr(query_list, doc_list) #填入VSM表格
print("vsm table complete")
print("vsm_arr = ", vsm_arr)
fileWrite(query_list, doc_list)
print("complete")