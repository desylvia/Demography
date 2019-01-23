import re
import nltk
import random
import pandas as pd
from names_clean import words_comp
from nltk.classify import apply_features


def reli_features(word): #fitur extraction (Islam, Kristen)
    features = {}
    features["first_letter"] = word[0].lower()
    features["last_letter"] = word[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = word.lower().count(letter)
        features["has({})".format(letter)] = (letter in word.lower())
    return features

#Load Data
islam = pd.read_csv('C:/_Religion/Islam.csv', encoding = "UTF-8", names=['Islam'])
kristen = pd.read_csv('C:/_Religion/kristen.csv', encoding = "UTF-8", names=['Kristen'])
hindu = pd.read_csv('C:/_Religion/Hindu.csv', encoding = "UTF-8", names=['Hindu']) #Hindu keys

#Islam n Kristen, naive bayes classifier training
list_kristen = []
for n in range(len(kristen)):
    list_kristen.append(kristen.Kristen[n].lower().strip())
list_islam = []
for n in range(len(islam)):
    list_islam.append(islam.Islam[n].lower().strip())

labeled_religion = ([(name,'islam') for name in list_islam] +
                  [(name,'kristen') for name in list_kristen])
random.shuffle(labeled_religion)

featuresets = [(reli_features(n),reli) for (n,reli) in labeled_religion]

train_set = apply_features(reli_features,labeled_religion)
classifier = nltk.NaiveBayesClassifier.train(train_set)

#Islam n Kristen, testing. Get first name from input data
names = []
for w in range(len(words_comp)):
    buff = words_comp[w].split()
    names.append(buff[0])

labels_data = []
for idx in range(len(names)):
    labels_data.append(classifier.classify(reli_features(names[idx]))) #testing process
labels_data = [[words_comp[idx], labels_data[idx]] for idx in range(len(names))]

#Hindu, key searching algorithm
for k in range(len(hindu)):
    p = re.compile(r'\b' + hindu.Hindu[k].lower().strip() + r'\b')

    for w in range(len(labels_data)):
        match = p.search(labels_data[w][0].lower().strip())
        if match:
            labels_data[w][1] = 'Hindu'

#Final result --> labels_data
labels = ['Nama','Religion']
labels_data = pd.DataFrame.from_records(labels_data, columns=labels)

#labels_data.to_csv('test_religion_3462data.csv', sep=';', encoding='utf-8', index = False)
