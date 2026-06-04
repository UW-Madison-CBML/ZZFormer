import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import load_model

# from sklearn.metrics import classification_report

import itertools
import sys
import re
import pickle

def word_seq(seq, k, stride=1):
    i = 0
    words_list = []
    while i <= len(seq) - k:
        words_list.append(seq[i: i + k])
        i += stride
    return (words_list)

def generate_kmer_dic(repeat_num):
    kmer_dic = {}
    bases = ['A','G','C','T']
    kmer_list = list(itertools.product(bases, repeat=int(repeat_num)))
    for eachitem in kmer_list:
        each_kmer = ''.join(eachitem)
        kmer_dic[each_kmer] = 0
    return (kmer_dic)

def generate_mat(words_list,kmer_dic):
    for eachword in words_list:
        kmer_dic[eachword] += 1
    num_list = []  ##this dic stores num_dic = [0,1,1,0,3,4,5,8,2...]
    for eachkmer in kmer_dic:
        num_list.append(kmer_dic[eachkmer])
    return (num_list)

def generate_mats(seqs):
    seq_mats = []
    for eachseq in seqs:
        words_list = word_seq(eachseq, 7, stride=1)  ##change the k to 3
        kmer_dic = generate_kmer_dic(7)  ##this number should be the same as the window slide number
        num_list = generate_mat(words_list,kmer_dic)
        seq_mats.append(num_list)
    return (seq_mats)





















def load_data(fname, n):
    seqs = []
    labels = []
    f = open(fname)
    for line in f:
        line_no_wspace = line.replace(" ","")
        line_no_nwline = line_no_wspace.replace("\n","")
        line_arr = line_no_nwline.split(",")
        label = line_arr[0].strip()
        seq = line_arr[1]
        seq = seq.upper()    # b/c rep matrix built on uppercase
        seq = seq.replace("\t","")      # present in promoter

        seq = seq.replace("Y","C")  # undetermined nucleotides in splice
        seq = seq.replace("D","G")
        seq = seq.replace("S","C")
        seq = seq.replace("R","G")
        seq = seq.replace("V","A")
        seq = seq.replace("U","C") # added U to C
        seq = seq.replace("K", "G")
        seq = seq.replace("N", "T")
        seq = seq.replace("H", "A")
        seq = seq.replace("W", "A")
        seq = seq.replace("M", "C")
        seq = seq.replace("X", "G")
        seq = seq.replace("B", "C")

        # all remaining alphabet characters -> G
        seq = seq.replace("E", "G")
        seq = seq.replace("F", "G")
        seq = seq.replace("I", "G")
        seq = seq.replace("J", "G")
        seq = seq.replace("L", "G")
        seq = seq.replace("O", "G")
        seq = seq.replace("P", "G")
        seq = seq.replace("Q", "G")
        seq = seq.replace("Z", "G")


        labels.append(label)
        seqs.append(seq)
    f.close()
    labels = [l.split('_')[n] if len(l.split('_')) >= n+1 else '' for l in labels]
    return seqs, labels

def remove_no_labels(labels, seqs):
    ls = []
    ss = []
    for l, s in zip(labels, seqs):
        if l == '':
            continue
        else:
            ls.append(l)
            ss.append(s)
    return ls, ss

def convert_labels(labels, id_dict):
    converted = [id_dict[l] for l in labels]
    return converted


K = 7
ALPHABET_SIZE = 4
VEC_LEN = ALPHABET_SIZE ** K   # 16384

train_file = sys.argv[1]
print(f"train_file: {train_file}")

test_file = sys.argv[2]
print(f"test_file: {test_file}")

n = int(sys.argv[3])
print(f"Level n: {n}")
save_outputdir = sys.argv[4]
print(f"save_outputdir: {save_outputdir}")
model_name = sys.argv[5]
print(f"model_name: {model_name}")

input_store_class_report_dir = save_outputdir

# Load data and process
# train_file = "./mntedb/fold_0_train_mntedb.txt"
# test_file = "./mntedb/fold_0_test_mntedb.txt"
train_X, train_y = load_data(train_file,n)
test_X, test_y = load_data(test_file,n)

train_y, train_X = remove_no_labels(train_y, train_X)
test_y, test_X = remove_no_labels(test_y, test_X)

train_X = generate_mats(train_X)
test_X = generate_mats(test_X)

input_data_nm = len(list(set(test_y)))

id_dict = {item: i for i, item in enumerate(sorted(set(train_y)))}
import json

with open(f"{input_store_class_report_dir}/{model_name}_label_map.json", "w") as f:
    json.dump(id_dict, f, indent=2, sort_keys=True)


train_y = convert_labels(train_y, id_dict)
test_y = convert_labels(test_y, id_dict)

# Convert to arrays
train_X = np.asarray(train_X)
test_X = np.asarray(test_X)

# train_y = np.asarray(train_y)
# test_y = np.asarray(test_y)

# Reshape
train_X = train_X.reshape(train_X.shape[0], 1, VEC_LEN, 1)
test_X = test_X.reshape(test_X.shape[0], 1, VEC_LEN, 1)

train_X = train_X.astype('float64')
test_X = test_X.astype('float64')

# Preprocess labels
class_num = len(id_dict)
Y_train_one_hot = np_utils.to_categorical(train_y, int(class_num))
Y_test_one_hot = np_utils.to_categorical(test_y, int(class_num))

# Define architecture
model = Sequential()

model.add(Conv2D(100, (1, 3), activation='relu', input_shape=(1, VEC_LEN, 1)))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Conv2D(150, (1, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Conv2D(225, (1, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(int(class_num), activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Fit model on training data
# model.fit(train_X, Y_train_one_hot, validation_data=(test_X, Y_test_one_hot),batch_size=32, epochs=10, verbose=1) 

model.fit(train_X, Y_train_one_hot, batch_size=32, epochs=10, verbose=1)




################################################ Evaluate ################################################

##########################################################################################################

# score = model.evaluate(test_X, Y_test_one_hot, verbose=1)
score = model.evaluate(test_X, Y_test_one_hot, verbose=1)
print("\nfinal validation score = " + str(score))

# print ("\nscore = " + str(score))

store_all_score_dic = {}
store_all_score_dic[input_data_nm] = str(score)
################################################ Save model ################################################
# 7.5.  save the model
model.save(input_store_class_report_dir + '/' + model_name + '_model.h5')

# classification report
predicted_classes = model.predict(test_X)
predicted_classes = np.argmax(predicted_classes, axis=1)



from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)

test_y = np.asarray(test_y, dtype=int)
predicted_classes = np.asarray(predicted_classes, dtype=int)

# Map indices back to class names
inv_map = {v: k for k, v in id_dict.items()}
target_names = [inv_map[i] for i in range(len(inv_map))]

# Macro summary
acc = accuracy_score(test_y, predicted_classes)
p, r, f1, _ = precision_recall_fscore_support(
    test_y, predicted_classes, average="macro", zero_division=0
)

macro_line = (
    f"Macro-Precision\t{p:.4f}\n"
    f"Macro-Recall\t{r:.4f}\n"
    f"Macro-F1\t{f1:.4f}\n"
    f"Accuracy\t{acc:.4f}\n"
)

report = classification_report(
    test_y, predicted_classes,
    target_names=target_names,
    digits=4,
    zero_division=0
)
cm = confusion_matrix(test_y, predicted_classes)

print(macro_line)
print(report)
print(cm)

with open(f"{input_store_class_report_dir}/{model_name}_class_report.txt", "w") as opt:
    opt.write(macro_line + "\n")
    opt.write(report + "\n\n")
    opt.write("Confusion matrix:\n")
    opt.write(np.array2string(cm) + "\n")



# gpus = tf.config.list_physical_devices('GPU')