import collections
import re

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


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
