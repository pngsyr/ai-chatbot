import nltk
import json
import pickle
import random
import numpy as np
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.model')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
     for i, word in sentence_words:
        if word == w:
            bag[i] = 1
    
    return np.array(bag)
def predict_class(sentence):
   bow = bag_of_words(sentence)
   res = model.predict(np.array([bow]))[0]
   ERROR_TRESHOLD = 0.25
   result = [[i, r] for i, r in enumerate(res) if r > ERROR_TRESHOLD]

   result.sort(key=lambda x: x[1], reverse=True)
   return_lists = []

   for r in result:
      return_lists.append({'intent': classes[r[0]],'probability': str(r[1])})
   return return_lists

def get_response(intents_list, intents_json):
   tag = intents_list[0]['intent']
   list_of_intents = intents_json['intents']
   for i in list_of_intents:
      if i['tag'] == tag:
         result = random.choice(i['responses'])
         break
      return result
   
print("GO! Bot is running")

while True:
   message = input("")
   ints = predict_class(message)
   res = get_response(ints, intents)
   print(res)
