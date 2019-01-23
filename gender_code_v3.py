import nltk
import random
import pandas as pd
from names_clean import words_comp
from nltk.classify import apply_features


def gender_features(word):
    return {'last_letter': word[-1]}


male = pd.read_csv('C:/_Gender/Male.csv', encoding = "UTF-8", names=['Male'])
female = pd.read_csv('C:/_Gender/Female.csv', encoding = "UTF-8", names=['Female'])

list_m = []
for n in range(len(male)):
    list_m.append(male.Male[n].lower().strip())

list_f = []
for n in range(len(female)):
    list_f.append(female.Female[n].lower().strip())

labeled_names = ([(name,'male') for name in list_m] +
                  [(name,'female') for name in list_f])
random.shuffle(labeled_names)

featuresets = [(gender_features(n),gender) for (n,gender) in labeled_names]

train_set = apply_features(gender_features,labeled_names)
classifier = nltk.NaiveBayesClassifier.train(train_set)

#Get first name
names = []
for w in range(len(words_comp)):
    buff = words_comp[w].split()
    names.append(buff[0])

labels_data = []
for idx in range(len(names)):
    labels_data.append(classifier.classify(gender_features(names[idx]))) #testing process
labels_data = [(words_comp[idx], labels_data[idx]) for idx in range(len(names))]

labels = ['Nama','Gender']
labels_data = pd.DataFrame.from_records(labels_data, columns=labels)

#labels_data.to_csv('test_gender_3462data.csv', sep=';', encoding='utf-8', index = False)
