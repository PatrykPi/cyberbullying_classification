import collections
import json
import re
import string
import time

import emoji
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from wordcloud import STOPWORDS
from wordcloud import WordCloud

STOPWORDS.update(['bc', 'n', 'm', 'im', 'll', 'y', 've', 'u', 'ur', 'don', 't', 's'])
model = spacy.load('en_core_web_sm')


def assemble():
   data_df = pd.read_csv("data/cyberbullying_tweets.csv")
   all_labels = data_df['cyberbullying_type'].unique()

   print('DUPLIKATY PRZED CLEANEM: ')
   print(data_df.duplicated().sum())
   print(data_df.cyberbullying_type.value_counts())
   data_df = data_df[~data_df.duplicated()]

   data_df['clean_tweets'] = clean_tweets_efficient(data_df['tweet_text'].tolist())
   print(data_df.sample(n=150))

   clean_dataset = data_df['clean_tweets']

   print('DUPLIKATY PO CLEANIE: ')
   print(clean_dataset.duplicated().sum())

   data_df.drop_duplicates("clean_tweets", inplace=True)
   print(data_df.cyberbullying_type.value_counts())

   data_df.to_csv(path_or_buf='data/output.csv', sep=';')

   print("Create CountVectorizer")

   vect = CountVectorizer(min_df=3)
   vect.fit(data_df['clean_tweets'])

   f = open("vocabulary.txt", "w")
   f.write(json.dumps(vect.vocabulary_))
   f.close()

   bag_of_words = vect.transform(data_df['clean_tweets'])

   X_train, X_test, y_train, y_test = train_test_split(bag_of_words, data_df['cyberbullying_type'])
   logreg = LogisticRegression(max_iter=300)
   logreg.fit(X_train, y_train)

   print(X_test[0])
   # print(y_test[0])
   y_pred = logreg.predict(X_test[0])
   print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
   print('predykcja: ', y_pred)
   # print("Cross validation")
   # scores = cross_val_score(LogisticRegression(max_iter=300), bag_of_words, data_df['cyberbullying_type'], cv=2)
   # print("Średnia dokładność walidacji krzyżowej: ")
   # print(scores)

   # gender_1 = pd.DataFrame(data_df['clean_tweets'], columns=['Text', 'count'])
   # gender_1.groupby('Text').sum()['count'].sort_values(ascending=True).iplot(
   #     kind='bar', yTitle='Count', linecolor='black', color='black', title='Top 10 Unigrams', orientation='h')

   # printCountWordPlot(data_df)
   # print("Prepare data set")
   # x, y = prepare_data()

   # print("Split into train and tast data set")
   # x_train, x_test, y_train, y_test = train_test_split(x, y)
   #

   #
   # print("Transform to bag of words")
   # x_train = vect.transform(x_train)
   #



# def cleanData(data):
#
#
# def remove_urls(text):
#     """Remove any URL/Hyperlink in the tweet"""
#     text = re.sub(r"(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*",
#                   "", text)
#     return text

def printLenghtPlotByClass(data_df, all_labels):
   fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 11))
   k = 0
   for i in range(3):
       for j in range(2):
           cat = all_labels[k]
           tweet_len = data_df[data_df['cyberbullying_type'] == cat]['tweet_text'].apply(lambda x: len(x))
           sns.histplot(tweet_len, ax=ax[i, j]);
           ax[i, j].set_title("Tweet Length distribution for '{}' Tweets".format(cat));
           ax[i, j].set_xlim(0, 350);
           plt.tight_layout();
           k += 1
   plt.show()


def printLenghtPlot(data_df):
   fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 11), squeeze=False)
   tweet_len = data_df['tweet_text'].apply(lambda x: len(x))
   sns.histplot(tweet_len, ax=ax[0, 0])
   ax[0, 0].set_title("Tweet Length distribution for Tweets");
   ax[0, 0].set_xlim(0, 350)
   plt.tight_layout()
   plt.show()


def printPunctuationPlot(data_df, all_labels):
   fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 11))
   k = 0
   for i in range(3):
       for j in range(2):
           cat = all_labels[k]
           tweet_punct = get_punct_count(data_df[data_df['cyberbullying_type'] == cat]['tweet_text'].tolist())
           tweet0_punct_cnt_sorted = tweet_punct.most_common()[1:]
           l0, h0, = [], []
           _ = [(l0.append(i[0]), h0.append(i[1])) for i in tweet0_punct_cnt_sorted]

           sns.barplot(x=list(range(len(l0))), y=h0, ax=ax[i, j]);
           ax[i, j].set_ylim(top=21000);
           ax[i, j].set_xticks(ticks=list(range(len(l0))), labels=l0);
           ax[i, j].set_xlabel('Punctuation');
           ax[i, j].set_ylabel('Count');
           ax[i, j].set_title("Punctuation Count for '{}' tweets".format(cat));
           k += 1
   fig.tight_layout();
   plt.show()


def get_punct_count(text_lst):
   punct_pattern = r"[!@#$%^&*()_\-=+}{\[\]|\\/<>,.?~`]?"
   all_puncts = list()
   for txt in text_lst:
       m_lst = re.findall(punct_pattern, txt)
       all_puncts.extend(m_lst)
   punct_counts = collections.Counter(all_puncts)
   return punct_counts


def printCountWordPlotByClass(data_df, all_labels):
   fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 11))
   k = 0
   for i in range(3):
       for j in range(2):
           cat = all_labels[k]
           # tweet_punct = get_wrd_count(data_df[data_df['cyberbullying_type']==cat]['tweet_text'].tolist())
           # tweet0_punct_cnt_sorted = tweet_punct.most_common()[1:]
           # l0, h0, = [],[]
           # _ = [(l0.append(i[0]), h0.append(i[1])) for i in tweet0_punct_cnt_sorted]
           wordcloud = WordCloud().generate(
               " ".join(data_df[data_df['cyberbullying_type'] == cat]['tweet_text'].tolist()))

           # Display the generated image:
           ax[i, j].imshow(wordcloud, interpolation='bilinear');
           # sns.barplot(x=list(range(len(l0))), y=h0, ax=ax[i,j]);
           # ax[i,j].set_ylim(top=21000);
           # ax[i,j].set_xticks(ticks = list(range(len(l0))), labels=l0);
           # ax[i,j].set_xlabel('Words');
           # ax[i,j].set_ylabel('Count');
           # ax[i,j].set_title("Word Count for '{}' tweets".format(cat));
           ax[i, j].set_title("Word Cloud for '{}' tweets".format(cat));
           k += 1
   fig.tight_layout();
   plt.show()


def printCountWordPlot(data_df):
   fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 11), squeeze=False)
   # tweet_punct = get_wrd_count(data_df[data_df['cyberbullying_type']==cat]['tweet_text'].tolist())
   # tweet0_punct_cnt_sorted = tweet_punct.most_common()[1:]
   # l0, h0, = [],[]
   # _ = [(l0.append(i[0]), h0.append(i[1])) for i in tweet0_punct_cnt_sorted]
   wordcloud = WordCloud().generate(
       " ".join(data_df['clean_tweets'].tolist()))

   # Display the generated image:
   ax[0, 0].imshow(wordcloud, interpolation='bilinear');
   # sns.barplot(x=list(range(len(l0))), y=h0, ax=ax[i,j]);
   # ax[i,j].set_ylim(top=21000);
   # ax[i,j].set_xticks(ticks = list(range(len(l0))), labels=l0);
   # ax[i,j].set_xlabel('Words');
   # ax[i,j].set_ylabel('Count');
   # ax[i,j].set_title("Word Count for '{}' tweets".format(cat));
   ax[0, 0].set_title("Word Cloud for tweets");
   fig.tight_layout();
   plt.show()


def clean_tweets_efficient(all_tweets):
   start = time.time()
   clean_tweets = list()
   for tweet in all_tweets:
       clean_level_1 = lower_case_words(tweet)
       clean_level_2 = clean_tags_mentions_single(clean_level_1)
       clean_level_3 = remove_urls(clean_level_2)
       clean_level_4 = remove_emojis_single(clean_level_3)

       # clean_level_4 = remove_single_chars(remove_remaining_punct(clean_level_2))
       clean_level_5 = remove_stopwords(clean_level_4)
       clean_level_6 = strip_punct_and_special_chars(clean_level_5)
       clean_level_7 = stemmer(clean_level_6)
       # clean_level_7 = lemmatize(clean_level_6)
       # clean_level_5 = correct_mispelled_words(clean_level_4)
       clean_tweets.append(clean_level_7)
   end = time.time()
   print("Time to clean {} tweets : {} seconds".format(len(all_tweets), end - start))
   return clean_tweets


def clean_tags_mentions_single(txt):
   patt_mention = r"[@]\w+"
   patt_tags = r"[#]\w+"
   clean_str = re.sub(patt_mention, "", txt)
   clean_str = re.sub(patt_tags, "", clean_str)
   clean_str = re.sub(r'[0-9]+', '', clean_str)
   clean_str = re.sub("'ve", " have ", clean_str)
   clean_str = re.sub("&amp;", "", clean_str)
   clean_str = re.sub("\n", "", clean_str)
   return " ".join(clean_str.split())


def stemmer(text):
   tokenized = nltk.word_tokenize(text)
   ps = PorterStemmer()
   return ' '.join([ps.stem(words) for words in tokenized])

def lemmatize(tweet_text):
   tokenized = nltk.word_tokenize(tweet_text)
   lm = WordNetLemmatizer()
   return ' '.join([lm.lemmatize(words) for words in tokenized])

def remove_stopwords(text):
   return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def strip_punct_and_special_chars(text):
   text = re.sub(r'[^\x00-\x7f]', r'', text)
   punct_list = string.punctuation
   for letter in text:
       if letter in punct_list:
           text = text.replace(letter, "")

   return "".join(text)


def lower_case_words(txt):
   lower_case_tweet = [word.lower() for word in txt]

   return "".join(lower_case_tweet)


def remove_emojis_single(txt):
   return emoji.replace_emoji(txt, replace='')


def remove_urls(text):
   """Remove any URL/Hyperlink in the tweet"""
   text = re.sub(r"(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*", "",
                 text)
   return text


if __name__ == '__main__':
   assemble()


