import glob
import numpy as np
import math
import os
import operator
import keras
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import  Adam
from keras.layers import Input, Dense, Embedding
from keras.models import Model



def create_BGLM():
	print("build dictionary")
	file = open(r"BGLM.txt")
	file = file.read()
	file_list = file.split("\n")
	key = []
	value = [] 
	for i in range(len(file_list)):
		key.append(file_list[i][0:file_list[i].find(" ")].strip())
		value.append(file_list[i][file_list[i].find(" "):].strip())
	key.pop()
	value.pop()

	return dict(zip(key, value)), key, value




if __name__ == "__main__":

	doc_list  = glob.glob(r'Document\*')
	doc_len = len(doc_list)
	BGLM_dic, BGLM_dic_key, BGLM_dic_value = create_BGLM()
	wordKeyList = BGLM_dic_key

	arr_onehot = np.zeros((1, 1, 51253), int)

	#defyine network
	Input_Left = Input(shape = (1, ))
	Input_Right = Input(shape = (1, ))
	Answer = Input(shape = (51253, )) #沒用到
	CBOW = Embedding(output_dim = 100, input_dim = 51253) #綠色部分
	CBOW_L = CBOW(Input_Left)
	CBOW_R = CBOW(Input_Right)

	Average = keras.layers.Average()([CBOW_L, CBOW_R])
	Prediction = Dense(51253, activation = "softmax")(Average)

	model = Model(inputs = [Input_Left, Input_Right], outputs = Prediction)
	print(model.summary())
	os.system("pause")
	model.compile(optimizer = "adam", loss = "categorical_crossentropy")

	for epoch in range(500):

		for i in range(doc_len):

			file = open(doc_list[i], 'r')
			file = file.read()
			file = file.split("\n")
			del file[0]
			del file[0]
			del file[0]
			file = ''.join(file)
			file = file.replace("-1", "")
			file = file.strip()
			file = file.split(" ")

			for j in range(len(file)):

				if j == len(file) - 2:
					break
				input_L = np.array(int(file[j])).reshape(1, 1)
				input_R = np.array(int(file[j + 2])).reshape(1, 1)
				arr_onehot[0, 0, int(file[j + 1])] = 1
				Answer_123 = arr_onehot	

				model.fit([input_L, input_R], Answer_123, batch_size = 1, verbose = 1)
				arr_onehot[0, 0, int(file[j + 1])] = 0

		model.save("model_" + str(epoch) + ".h5")

	model.save("model.h5")