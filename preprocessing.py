import re
import string
import time

import emoji
import nltk
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from wordcloud import STOPWORDS


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
