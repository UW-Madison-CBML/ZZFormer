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
def generate_kmer_dic (repeat_num):
    kmer_dic = {}
    bases = ['A','G','C','T']
    kmer_list = list(itertools.product(bases, repeat=int(repeat_num)))
    for eachitem in kmer_list:
        each_kmer = ''.join(eachitem)
        kmer_dic[each_kmer] = 0
    return (kmer_dic)
def generate_mat (words_list,kmer_dic):
    for eachword in words_list:
        kmer_dic[eachword] += 1
    num_list = []  ##this dic stores num_dic = [0,1,1,0,3,4,5,8,2...]
    for eachkmer in kmer_dic:
        num_list.append(kmer_dic[eachkmer])
    return (num_list)
def generate_mats (seqs):
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
        label = line_arr[0]
        seq = line_arr[1]
        seq = seq.upper()    # b/c rep matrix built on uppercase
        seq = seq.replace("\t","")      # present in promoter
        seq = seq.replace("Y","C")  # undetermined nucleotides in splice
        seq = seq.replace("D","G")
        seq = seq.replace("S","C")
        seq = seq.replace("R","G")
        seq = seq.replace("V","A")
        seq = seq.replace("K", "G")
        seq = seq.replace("N", "T")
        seq = seq.replace("H", "A")
        seq = seq.replace("W", "A")
        seq = seq.replace("M", "C")
        seq = seq.replace("X", "G")
        seq = seq.replace("B", "C")
        labels.append(label)
        seqs.append(seq)
    f.close()
    labels = [l.split('_')[n] for l in labels]
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


# train_file = sys.argv[1]
# test_file = sys.argv[2]
# n = int(sys.argv[3])

# input_store_class_report_dir = './outputs/'



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

train_y = convert_labels(train_y, id_dict)
test_y = convert_labels(test_y, id_dict)

# Convert to arrays
train_X = np.asarray(train_X)
test_X = np.asarray(test_X)

# train_y = np.asarray(train_y)
# test_y = np.asarray(test_y)

# Reshape
train_X = train_X.reshape(train_X.shape[0], 1, 16384, 1)
test_X = test_X.reshape(test_X.shape[0], 1, 16384, 1)

train_X = train_X.astype('float64')
test_X = test_X.astype('float64')

# Preprocess labels
class_num = len(id_dict)
Y_train_one_hot = np_utils.to_categorical(train_y, int(class_num))
Y_test_one_hot = np_utils.to_categorical(test_y, int(class_num))

# Define architecture
model = Sequential()

model.add(Conv2D(100, (1, 3), activation='relu', input_shape=(1, 16384, 1)))
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
model.fit(train_X, Y_train_one_hot, validation_data=(test_X, Y_test_one_hot),batch_size=32, epochs=10, verbose=1) 

# Evaluate
score = model.evaluate(test_X, Y_test_one_hot, verbose=1)
print ("\nscore = " + str(score))

store_all_score_dic = {}
store_all_score_dic[input_data_nm] = str(score)

# Save model

# classification report
predicted_classes = model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

#print('predicted_classes is ')
#print(predicted_classes)
#print('test_Y is ')
#print(Y_test)

##change the Y_test to array
##Y_test is a list
test_y = np.asarray(test_y)
correct = np.where(predicted_classes==test_y)[0]

results = {"score": store_all_score_dic,
           "pred_class": predicted_classes,
           "test_y": test_y,
           "correct": correct,
           "maps": id_dict}
test_name= test_file.split('/')[-1]
save = test_name.replace('.txt', '.pkl')
with open(f"{save_outputdir}/{save}", 'wb') as file:
    pickle.dump(results, file)

# with open(f"{save_outputdir}/{save}", 'rb') as file:
#     data = pickle.load(file)

# print(data)

print ("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:int(class_num)]):
    #plt.subplot(3,3,i+1)
    #plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
    print("Predicted {}, Class {}".format(predicted_classes[correct], test_y[correct]))


# target_names = ["Class {}".format(i) for i in range(int(class_num))] ##there are four classes

# with open (input_store_class_report_dir + '/' + input_data_nm + '_class_report.txt','w+') as opt:
#     opt.write(classification_report(test_y, predicted_classes, target_names=target_names) + '\n')



#gpus = tf.config.list_physical_devices('GPU')


import pickle
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support,
)

with open(f"{save_outputdir}/{save}", "rb") as f:
    r = pickle.load(f)

y_true = np.asarray(r["test_y"], dtype=int)
y_pred = np.asarray(r["pred_class"], dtype=int)

# id -> class name (inverse of r["maps"])
inv_map = {v: k for k, v in r["maps"].items()}
target_names = [inv_map[i] for i in range(len(inv_map))]

# Macro metrics
acc = accuracy_score(y_true, y_pred)
p, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)
print(f"Accuracy      : {acc:.4f}")
print(f"Macro-Precision: {p:.4f}")
print(f"Macro-Recall   : {rec:.4f}")
print(f"Macro-F1       : {f1:.4f}")

# Per-class breakdown
print(classification_report(
    y_true, y_pred, target_names=target_names, digits=4, zero_division=0
))

# Confusion matrix (rows = true, cols = pred), in the same order as target_names
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))

with open(f"{input_store_class_report_dir}/{model_name}_class_report.txt", "w") as opt:
    opt.write(f"Accuracy      : {acc:.4f}\n")
    opt.write(f"Macro-Precision: {p:.4f}\n")
    opt.write(f"Macro-Recall   : {rec:.4f}\n")
    opt.write(f"Macro-F1       : {f1:.4f}\n\n")
    opt.write(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0) + "\n")
    opt.write("Confusion matrix:\n")
    opt.write(np.array2string(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))) + "\n")
