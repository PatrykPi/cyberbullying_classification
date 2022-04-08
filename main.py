import json

import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from wordcloud import STOPWORDS

from preprocessing import clean_tweets_efficient

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
    logreg = LogisticRegression(max_iter=300, C=0.25)
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

    # param_grid = {'C': [0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5]}
    # grid = GridSearchCV(LogisticRegression(max_iter=300), param_grid, cv=2)
    # grid.fit(X_train, y_train)
    # print("Najlepszy wynik walidacji krzyżowej: {:.2f}".format(grid.best_score_))
    # print("Najlepsze parametry: ", grid.best_params_)

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


if __name__ == '__main__':
    assemble()
