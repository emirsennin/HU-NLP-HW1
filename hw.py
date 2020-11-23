import sys
import os
import nltk.data
from nltk import ngrams

file = open('./hw01_tiny.txt')
print(os.path.basename(file.name))

data = file.read()

sentenceTokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentenceTokens = sentenceTokenizer.tokenize(data)
print("Number of Sentences in Test File : " + str(len(sentenceTokens)))

wordTokens = nltk.word_tokenize(data)
wordTokens = [word.lower() for word in wordTokens if word.isalpha()]
print("Number of Total Tokens : " +str(len(wordTokens)))

fdist1 = nltk.FreqDist(wordTokens)
filtered_word_freq = dict((word, freq) for word, freq in fdist1.items() if not word.isdigit())
print("Number of Unique Words (Vocabulary Size) : " + str(len(filtered_word_freq.values())))

for n in range(1,3):
    #while n = 1 it works as unigram, while n = 2 it works as bigram.
    for sentence in sentenceTokens:
        unigrams = ngrams(sentence.split(),n)

        for grams in unigrams:
            print(grams)
#https://stackoverflow.com/questions/54962539/how-to-get-the-probability-of-bigrams-in-a-text-of-sentences


print('\n-----\n'.join(sentenceTokenizer.tokenize(data)))