'''
Author : Wonhee Jung ( wonheej2@illinois.edu, wonhee.jung@gmail.com )
Since : Nov, 2018

UIUC MCS-DS CS410 Fall 2018 Project.
'''
import csv
import gc
import os
import os.path
import sys
import time
from builtins import range

import nltk
import numpy as np
import pandas as pd
from joblib import dump, load
from nltk import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix

# includes all the defined constant variables.
import constant

os.environ['OMP_NUM_THREADS'] = '2'

# The class will create two joblib files to store the trained classifier and vectorizer so that it can be reused quickly
# instead of training it from the beginning.
# 
# !!!! IMPORTANT !!!
#
# Which means if you ever chagne any code in __init__ that could affect the way how classifier/vectorizer is going to work,
# then you need to delete two files so that program recrete them. Those two files you need to delete are "classifier.joblib", and "vectorizer.joblib"
# defined in constant.py.
class ToxicityClassifier():
    def __init__(self):

        start_time = time.time()

        self.stopwords = set(w.rstrip() for w in open('stopwords.txt'))
        self.vectorizer = TfidfVectorizer(tokenizer=ToxicityClassifier.tokenizer, max_features=10000, stop_words=self.stopwords, analyzer='word', dtype=np.float32)

        # creating the Multinomial Naive Bayes with Laplacian smoothing.
        self.classifier = MultinomialNB()

        # if and only if model doesn't exist in the file, execute this block, means you need to delete the existing model file to re-run this.
        if os.path.exists(constant.CLASSIFIER_FILE) == False:
            print("Can't find existing classifier stored in the file. Creating one...")

            # for df in pd.read_csv(constant.TRAINING_DATA_PATH, delimiter=',', error_bad_lines=True, skipinitialspace=True, chunksize=constant.CVS_CHUNKSIZE, iterator=True):
            dtypes = {
                'toxic' : 'uint8',
                'severe_toxic' : 'uint8',
                'obscene' : 'uint8',
                'threat' : 'uint8',
                'insult' : 'uint8',
                'identity_hate' : 'uint8'
            }

            df = pd.read_csv(constant.TRAINING_DATA_PATH, dtype=dtypes)
            df = df.replace('\n','', regex=True)

            toxic = \
                df['toxic'] | \
                df['severe_toxic'] | \
                df['obscene'] | \
                df['threat'] | \
                df['insult'] | \
                df['identity_hate'] 
            
            training = self.vectorizer.fit_transform(df['comment_text'])  
            del df
            gc.collect()

            print("Initiating training...")
            train_x, test_x, train_y, test_y = train_test_split(training.toarray(), toxic.values, test_size=0.2)
            del training
            gc.collect()
            self.classifier.fit(train_x, train_y)

            del train_x, train_y
            gc.collect()
                
            print("Completed training. Generating classification result...")
            pred_y = self.classifier.predict(test_x)

            del test_x
            gc.collect()

            print(classification_report(test_y, pred_y))

            # store the classifier and vectorizer so it can be used later    
            print("Storing classifier and vectorizer into disk...")
            dump(self.classifier, constant.CLASSIFIER_FILE)
            dump(self.vectorizer, constant.VECTORIZER_FILE)
        else:
            print("Found existing classifier and vectorizer stored in the file. Loading...")
            self.classifier = load(constant.CLASSIFIER_FILE)
            self.vectorizer = load(constant.VECTORIZER_FILE)

        # to measure programing execution time
        print("--- time spent for initializing the classifier : %s seconds ---" % (time.time() - start_time))


    # get string value as paraemter and return tokenized array after some data massage, cleansing, etc.
    # returned token array won't have any word with length of one or two
    @staticmethod
    def tokenizer(s):
        wordnet_lemmatizer = WordNetLemmatizer()

        s = s.lower() 
        tokens = nltk.tokenize.word_tokenize(s) 
        tokens = [token for token in tokens if len(token) > 2] # remove words with two characters or less, those are probably pronoun and 
        tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens] # change the word to base form with NLTK's WordNetLematizer
        return tokens

    # with given parameter s, it returns whether s is toxic or not
    # it is not expecting any arrays, it should be just single string value
    def isToxic(self, s):

        pred = self.classifier.predict( self.vectorizer.transform( np.array([s])).toarray() )
        
        if pred[0] == 1:
            return True
        else:
            return False

# main function if you need to run it separated, not through chatbot.py.
# The function will load local test CSV file and execute the prediction, instead of getting messages from TwitchTV channel it has deployed.
# With "test/test.csv", the code is going to take anothe 300MB +- to load all the text data and process it.
def main():
    print("Initiating...")

    # below file has smaller set of test data. Enable it instead if you want quicker testing
    test_data_path = "data/test_sample.csv"
    # below file has full set of test data. Try with it if you see more dresult. Beware : it will take some time.
    #test_data_path = "data/test.csv"

    toxicClassifier = ToxicityClassifier()

    # loading chat logs from csv file
    df = pd.read_csv(test_data_path, delimiter=',', error_bad_lines=True, skipinitialspace=True)
    df = df.replace('\n','', regex=True)

    # transform test data's chat log to the existing vecorizer so it can be used for prediction        
    data = toxicClassifier.vectorizer.transform( df['comment_text'] )
    preds = toxicClassifier.classifier.predict( data )

    # print(pd.DataFrame(preds, columns=toxicClassifier.classifier.classes_))
    for i in range(len(preds)):
        if preds[i] == 1:
            print("TOXIC>>> " + df['comment_text'].values[i])            


# just in case if need to run the test locally without TwitchBox working together
if __name__ == "__main__":

    if not sys.version_info[:1] == (3,):
        print(sys.version_info[:1] )
        sys.stderr.write("Python version 3 is required.\n")
        exit(1)

    main()
