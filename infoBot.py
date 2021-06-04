import random
import os
import json
import pickle
import numpy as np
import pandas as pd
import csv
import itertools
import nltk
import sys
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from yaml import cyaml
from collections import defaultdict
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

nltk.download('averaged_perceptron_tagger')

intents = json.loads(
    open('/home/jacob/MachineLearning/Chatbots/infotents.json').read())

lastChecked = ""
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']
csv_columns = ['name','favorite color', 'favorite food', 'favorite animal', 'least favorite color', 'least favorite food', 'least favorite animal']
mem_dict = {}
with open("Bot_Memeory.csv", 'r+') as data_file:
    data_file.readline()
    for row in data_file:
        row = row.strip().split(",")
        mem_dict[row[0]] = {'favorite color': row[1], 'favorite food': row[2], 'favorite animal': row[3], 'least favorite color': row[4], 'least favorite food': row[5], 'least favorite animal': row[6]}
    print(mem_dict)
currentIndex = 'Jacob'


model = load_model('chatbotmodel.h5')

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word)
         for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append(classes[r[0]])
    return return_list


def tokenize_sentence(sentType, item):
    sentences = nltk.sent_tokenize(item)  # tokenize sentences
    nouns = []  # empty to array to hold all nouns
    adjectives = []
    properNouns = []

    for sentence in sentences:
        for word, pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
            if(pos == 'NNP'):
                properNouns.append(word)
            if (pos == 'NN' or pos == 'NNS' or pos == 'NNPS'):
                nouns.append(word)
            if (pos == 'JJ' or pos == 'JJR' or pos == 'JJS' or pos == 'NNPS'):
                adjectives.append(word)

    if(sentType == 'noun'):
        return nouns
    if(sentType == 'adjective'):
        return adjectives
    if(sentType == 'proper noun'):
        return properNouns


def getObj(message, name, tag, wordType):
    term = tokenize_sentence(wordType, message)
    lterm = term[-1]
    #print(tag[-1])
    #print(mem_dict[currentIndex])
    mem_dict[currentIndex][tag[-1]] =  lterm



def getName(wordType, message):
    name = tokenize_sentence(wordType, message)
    lname = name[0]
    return lname


print("bot is running")

while True:
    message = input("say something: ")

    messageClass = predict_class(message)

    if(messageClass == ['greetings']):
        print('hello')

    if(messageClass == ['goodbye']):
        print('bye')
        csv_file = "Bot_Memeory.csv"
        with open(csv_file, "w", newline='') as f:
            w = csv.DictWriter(f, csv_columns)
            w.writeheader()
            for k in mem_dict:
                w.writerow({field: mem_dict[k].get(field) or k for field in csv_columns})

        sys.exit()

    if(messageClass == ['name']):
        name = getName('proper noun', message)
        #d = {'color', 'food', 'animal'}
        currentIndex = name
        if name in mem_dict:
            print(f'welcome back {name}')
            
        else:
            mem_dict[name] = {'favorite color':'_', 'favorite food':'_', 'favorite animal':'_', 'least favorite color':'_', 'least favorite food':'_', 'least favorite animal':'_'}
            print(mem_dict)
            
            print(currentIndex)

    if(messageClass == ['favorite food']):
        getObj(message, currentIndex, messageClass, 'noun')

    if(messageClass == ['favorite animal']):
        getObj(message, currentIndex, messageClass, 'noun')

    if(messageClass == ['favorite color']):
        getObj(message, currentIndex, messageClass, 'adjective')
        
    if(messageClass == ['least favorite food']):
        getObj(message, currentIndex, messageClass, 'noun')

    if(messageClass == ['least favorite animal']):
        getObj(message, currentIndex, messageClass, 'noun')

    if(messageClass == ['least favorite color']):
        getObj(message, currentIndex, messageClass, 'adjective')
    #inquiry 
    if(random.randrange(1,3) == 2):
        rvalue = random.choice(csv_columns)
        if(mem_dict[currentIndex][rvalue] != '_' and mem_dict[currentIndex][rvalue] != lastChecked):
            inquiry = input(f"is your {rvalue} still {mem_dict[currentIndex][rvalue]}?: ")
            responce = predict_class(inquiry)
            if(responce == ['yes']):
                print("thanks for confirming")
            
            if(responce == ['no']):
                nvalue = input("what is it?: ")
                for intent in intents['intents']:
                    if(intent['tag'] == rvalue):
                        term = tokenize_sentence(intent['type'], nvalue)
                        lterm = term[-1]
                        mem_dict[currentIndex][rvalue] =  lterm
        lastChecked = rvalue
        

