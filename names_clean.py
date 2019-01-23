import re
import pandas as pd


raw_data = pd.read_csv('C:/_Testing/test_gender_religion.csv', encoding = "UTF-8", sep="\n")
#raw_data = pd.read_csv('C:/_Testing/test_gender_religion_2.csv', encoding = "UTF-8", sep=";")
my_data = raw_data.dropna().reset_index(drop=True)

name = my_data.name

words_comp = []
for w in range(len(name)):
    words_comp.append(re.sub('[^a-zA-Z ]','',str(name[w].lower().strip())))

test = []
for w in range(len(words_comp)):
    test.append(re.sub('  +','',words_comp[w].strip()))

words_comp = [x for x in test if x]