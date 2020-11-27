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
    # print('\n-----\n'.join(sentenceTokenizer.tokenize(data)))
    return sentenceTokens

def wordTokenize(data):
    wordTokens = nltk.word_tokenize(data)
    wordTokens = [word.lower() for word in wordTokens if word.isalpha()]
    return wordTokens

def freqFinder(wordTokens):
    fdist1 = nltk.FreqDist(wordTokens)
    filtered_word_freq = dict((word, freq) for word, freq in fdist1.items() if not word.isdigit())
    return filtered_word_freq

def unigramFinder(sentenceTokens,wordCount):
    unigramList = []
    for sentence in sentenceTokens:
        tokenizer = RegexpTokenizer(r'\w+')
        new_sentence = " ".join(tokenizer.tokenize(sentence)).lower()
        unigrams = ngrams(new_sentence.split(), 1)
        for grams in unigrams:
            unigramList.append(" ".join(grams))
    count = Counter(unigramList)
    unigramDf = pd.DataFrame.from_dict(count, orient='index').reset_index()
    unigramDf = unigramDf.rename(columns={'index':'Text',0:'Count'})
    unigramDf["Probability"] = np.nan
    for index,row in unigramDf.iterrows():
        # row.Probabilty = row.Count/wordCount
        unigramDf.iloc[index, unigramDf.columns.get_loc('Probability')] = row.Count/wordCount

    return unigramDf

def bigramFinder(sentenceTokens,unigramDf,k,unique_vocab):
    bigramList = []
    for sentence in sentenceTokens:
        tokenizer = RegexpTokenizer(r'\w+')
        new_sentence = " ".join(tokenizer.tokenize(sentence)).lower()
        new_sentence = "<s> "+ new_sentence + " </s>"
        bigrams = ngrams(new_sentence.split(), 2)

        for grams in bigrams:
            bigramList.append(" ".join(grams))
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
                (row.Count + k) / (unigramDf.loc[unigramDf["Text"] == bigramDf.Text[index].split()[0]].Count.item() + (k * unique_vocab))

    # bigramProbality = ikilinin unigram countu/ilk kelimenin unigram countu
    return bigramDf


def makeLowestUnknown(data,unigramDf):
    replaces = unigramDf.sort_values(by="Probability",ascending=False).Text.values[-3:]
    newData = data.replace(replaces[0],"UNK").replace(replaces[1],"UNK").replace(replaces[2],"UNK")
    return newData

def calculateProbabilityOfNewSentence():
    pass

def generalFunc(path,name):
    print("\n\n--------- Starting To Work For "+name+"-------------\n\n")
    file = open(path)
    writeFile = open(name+"_Result.txt","a")
    writeFile.writelines(str(os.path.basename(file.name) + "\n\n"))

    data = file.read()

    sentenceTokens = sentenceTokenize(data)
    writeFile.writelines("Number of Sentences in Test File : " + str(len(sentenceTokens))+"\n\n")

    wordTokens = wordTokenize(data)
    writeFile.writelines("Number of Total Tokens : " + str(len(wordTokens))+"\n\n")

    filtered_word_freq = freqFinder(wordTokens)
    writeFile.writelines("Number of Unique Words (Vocabulary Size) : " + str(len(filtered_word_freq.values()))+"\n\n")

    unigramDf = unigramFinder(sentenceTokens,len(wordTokens))
    writeFile.writelines("Top 10 Unigrams with Highest Frequencies:\n\n")
    writeFile.writelines(unigramDf.sort_values(by="Probability",ascending=False).head(10).to_string()+"\n\n")

    bigramDf = bigramFinder(sentenceTokens,unigramDf,0,0)
    writeFile.writelines("Top 10 Bigrams with Highest Frequencies:\n\n")
    writeFile.writelines(bigramDf.sort_values(by="BiProbability", ascending=False).head(10).to_string()+"\n\n")

    writeFile.writelines("After UNK addition and Smoothing Operations:\n\n")

    data_with_unknown = makeLowestUnknown(data,unigramDf)

    sentenceTokensWitUnknown = sentenceTokenize(data_with_unknown)

    wordTokensWithUnknown = wordTokenize(data_with_unknown)

    filtered_word_freq_unknown = freqFinder(wordTokensWithUnknown)

    unigramDfWithUnknown = unigramFinder(sentenceTokensWitUnknown,len(wordTokensWithUnknown))

    bigramDfWithUnknown = bigramFinder(sentenceTokensWitUnknown,unigramDfWithUnknown,0,0)
    writeFile.writelines("Top 10 Bigrams with Highest Frequencies:\n\n")
    writeFile.writelines(bigramDfWithUnknown.sort_values(by="BiProbability", ascending=False).head(10).to_string()+"\n\n")

    smoothedBigramDf = bigramFinder(sentenceTokensWitUnknown,unigramDfWithUnknown,0.5,len(filtered_word_freq_unknown))
    writeFile.writelines("Top 10 Bigrams with UNK:\n\n")
    writeFile.writelines(smoothedBigramDf.sort_values(by="BiProbability", ascending=False).head(10).to_string()+"\n\n")
    writeFile.close()
    # new_sentence = input("Enter your sentence: ")
    # newSentenceTokens = sentenceTokenize(new_sentence)
    # newSentenceWordTokens = wordTokenize(new_sentence)
    # newSentenceUnigram = unigramFinder(newSentenceTokens,len(newSentenceWordTokens))
    # newSentenceBigram = bigramFinder(newSentenceTokens,newSentenceUnigram,0,0)
    #
    # calculateProbabilityOfNewSentence()

    print("\n\n--------- Finished Working For "+name+"-------------\n\n")



if __name__ == "__main__":
    pathList = ['./hw01_tiny.txt','./hw01_FireFairies.txt','./hw01_AMemorableFancy.txt']
    nltk.download('punkt')
    for path in pathList:
        name = path[2:-4]
        generalFunc(path,name)




    #https://stackoverflow.com/questions/54962539/how-to-get-the-probability-of-bigrams-in-a-text-of-sentences


