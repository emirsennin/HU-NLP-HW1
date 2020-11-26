import sys
import os
import nltk.data
from nltk import ngrams
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import pandas as pd
import numpy as np

def sentenceTokenize(data):
    sentenceTokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentenceTokens = sentenceTokenizer.tokenize(data)
    print("Number of Sentences in Test File : " + str(len(sentenceTokens)))
    print('\n-----\n'.join(sentenceTokenizer.tokenize(data)))
    return sentenceTokens

def wordTokenize(data):
    wordTokens = nltk.word_tokenize(data)
    wordTokens = [word.lower() for word in wordTokens if word.isalpha()]
    print("Number of Total Tokens : " + str(len(wordTokens)))
    return wordTokens

def freqFinder(wordTokens):
    fdist1 = nltk.FreqDist(wordTokens)
    filtered_word_freq = dict((word, freq) for word, freq in fdist1.items() if not word.isdigit())
    print("Number of Unique Words (Vocabulary Size) : " + str(len(filtered_word_freq.values())))
    return filtered_word_freq

def unigramFinder(sentenceTokens,wordCount):
    unigramList = []
    for sentence in sentenceTokens:
        tokenizer = RegexpTokenizer(r'\w+')
        new_sentence = " ".join(tokenizer.tokenize(sentence)).lower()
        unigrams = ngrams(new_sentence.split(), 1)
        for grams in unigrams:
            unigramList.append(" ".join(grams))
            print(grams)
    count = Counter(unigramList)
    unigramDf = pd.DataFrame.from_dict(count, orient='index').reset_index()
    unigramDf = unigramDf.rename(columns={'index':'Text',0:'Count'})
    unigramDf["Probability"] = np.nan
    for index,row in unigramDf.iterrows():
        # row.Probabilty = row.Count/wordCount
        unigramDf.iloc[index, unigramDf.columns.get_loc('Probability')] = row.Count/wordCount

    return unigramDf

def bigramFinder(sentenceTokens,unigramDf):
    bigramList = []
    for sentence in sentenceTokens:
        tokenizer = RegexpTokenizer(r'\w+')
        new_sentence = " ".join(tokenizer.tokenize(sentence)).lower()
        new_sentence = "<s> "+ new_sentence + " </s>"
        bigrams = ngrams(new_sentence.split(), 2)

        for grams in bigrams:
            bigramList.append(" ".join(grams))
            print(grams)
    count = Counter(bigramList)
    bigramDf = pd.DataFrame.from_dict(count, orient='index').reset_index()
    bigramDf = bigramDf.rename(columns={'index':'Text',0:'Count'})
    bigramDf["UniProbabilty"] = np.nan
    bigramDf["BiProbability"] = np.nan

    for index,row in bigramDf.iterrows():
        bigramDf.iloc[index, bigramDf.columns.get_loc('UniProbabilty')] = row.Count/len(bigramList)

    item = unigramDf.loc[unigramDf["Text"] == bigramDf.Text[0].split()[1]].Probability.item()
    for index,row in bigramDf.iterrows():
        if bigramDf.Text[index].split()[0] == "<s>":
            bigramDf.iloc[index, bigramDf.columns.get_loc('BiProbability')] = \
                unigramDf.loc[unigramDf["Text"] == bigramDf.Text[0].split()[1]].Probability.item()
        else:
            bigramDf.iloc[index, bigramDf.columns.get_loc('BiProbability')] = \
                row.Count/ unigramDf.loc[unigramDf["Text"] == bigramDf.Text[index].split()[0]].Count.item()

    # bigramProbality = ikilinin unigram countu/ilk kelimenin unigram countu
    return bigramDf


def generalFunc(path,name):
    print("\n\n--------- Starting To Work For "+name+"-------------\n\n")
    file = open(path)
    print(os.path.basename(file.name))

    data = file.read()

    sentenceTokens = sentenceTokenize(data)

    wordTokens = wordTokenize(data)

    filtered_word_freq = freqFinder(wordTokens)

    unigramDf = unigramFinder(sentenceTokens,len(wordTokens))

    bigramDf = bigramFinder(sentenceTokens,unigramDf)
    print(bigramDf.sort_values(by="BiProbability", ascending=False))
    print(bigramDf.head())

    print("\n\n--------- Finished Working For "+name+"-------------\n\n")



if __name__ == "__main__":
    pathList = ['./hw01_tiny.txt','./hw01_FireFairies.txt','./hw01_AMemorableFancy.txt']
    for path in pathList:
        name = path[2:-4]
        generalFunc(path,name)




    #https://stackoverflow.com/questions/54962539/how-to-get-the-probability-of-bigrams-in-a-text-of-sentences


