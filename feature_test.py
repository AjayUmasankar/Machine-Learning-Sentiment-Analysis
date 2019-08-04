'''
Created on 29 Jul 2019

@author: Ajay
'''
import re
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# So to make it simple (not necessarily the best), you can keep #, @, _, $ or % as
# part of words, but remove all other characters 
# (this is for the machine learning models, not VADER).
# @#$%_
# so no ' as in isn't -> isnt

# sentence_array = ['@TonyAbbottMHR forgets he was born in England #Brexit #ausvotes  https://t.co/DOUFLAajtR',
#                       '#ausvotes Insulting migrants in a country that imports 300,000 p.a. isn\'t good policy, but when has the LNP ever had a good policy?',
#                       'Government underestimates numbers affected by super changes new survey results show @cpaaustralia #ausvotes #auspol https://t.co/Z8FpBzPQMD',
#                       'Why does an education Australia threaten the LNP? @ABCFactCheck #auspol',
#                       '.#auspol .#ausvotes @wrst500 @australian Labor are fiscal simpletons - they have no experience with "real money", only debt!']
# sentence_array = ["at the very end https://t.co/QgHtSyva3Q",
#                     "https://t.co/QgHtSyva3Q beginning boy",
#                     "and in the https://t.co/QgHtSyva3Q middle ",
#                     "should $ not # be @ removed _ ever % ok?",
#                     "should , be \" removed ! at : all ; costs . () ",
#                     "$$moneymoneyca $1,083,000,000cash"]

# Create bag of words
count = CountVectorizer(token_pattern='[^\s]+', lowercase=False)

sentence_array = ['dont_ remove@asd %any of th_#@$%ese onebill$$ion$$ howdy freedom bu test TEST']
for i, sentence in enumerate(sentence_array):
    sentence = re.sub("http.*?(\s|$)", " ",sentence)
    sentence = re.sub("[.\\\,<>\[\]\|\/!\^&‘’\*;:{}=\-\'`~()\"”“,+\?—…–]", "", sentence) # $, #, @, _, %
    sentence_array[i] = sentence
text_data = np.array(sentence_array)
bag_of_words = count.fit_transform(text_data)
print(count.get_feature_names())


sentence_array1 = ['remo!ve all TE!"&\'()*+,-ST of these test',
                   'and  ./:;<=>? these',
                   'these [\]^`{|}~ too',
                   '"""unions""" brother']
for i, sentence in enumerate(sentence_array1):
    sentence = re.sub("http.*?(\s|$)", " ",sentence)
    sentence = re.sub("[.\\\,<>\[\]\|\/!\^&‘’\*;:{}=\-\'`~()\"”“,+\?—…–]", "", sentence) # $, #, @, _, %
    sentence_array1[i] = sentence
text_data = np.array(sentence_array1)
bag_of_words = count.fit_transform(text_data)
print(count.get_feature_names())

