'''
Author : Wonhee Jung, Cindy Tseng ( wonheej2@illinois.edu, wonhee.jung@gmail.com, cindyst2@illinois.edu )
Since : Nov, 2018

UIUC MCS-DS CS410 Fall 2018 Project.
'''

import gc
import os
import os.path
import sys
import time
import numpy as np
from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
from pyspark.sql.types import StringType
#from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import NaiveBayes
from pyspark.sql.functions import col, expr, when
from pyspark.ml import Pipeline, PipelineModel
#from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
# includes all the defined constant variables.
#from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

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
        
        # start spark session
        self.spark = SparkSession.builder.appName("DetoxBot").getOrCreate()
        self.sc = self.spark.sparkContext
        self.sqlContext = SQLContext(self.sc)
        self.stopwords = list(set(w.rstrip() for w in open('stopwords.txt')))

        # if and only if model doesn't exist in the file, execute this block, means you need to delete the existing model file to re-run this.
        if os.path.exists(constant.MODEL_DIR) == False:
            print("Can't find existing classifier stored in the file. Creating one...")
            
            # read data
            spDF = self.sqlContext.read.csv(constant.TRAINING_DATA_PATH, 
                                       header="true", 
                                       multiLine=True, 
                                       inferSchema=True,
                                       escape="\"")
            
            # generate label column
            spDF = spDF.withColumn('label', 
                spDF['toxic'] + \
                spDF['severe_toxic'] +  \
                spDF['obscene'] + \
                spDF['threat'] + \
                spDF['insult'] + \
                spDF['identity_hate'] )
            spDF = spDF.withColumn('label', when(spDF['label'] > 0, 1).otherwise(0) )
                
            # generate features column
            # tokenize
            tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
            wordsData = tokenizer.transform(spDF)
            
            # stop words remover
            remover = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords=self.stopwords)
            filteredData = remover.transform(wordsData)
            
            # term frequency transformation
            hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures")
            featurizedData = hashingTF.transform(filteredData)
            
            # inverse document frequency transformations
            idf = IDF(inputCol="rawFeatures", outputCol="features")
            idfModel = idf.fit(featurizedData)
            rescaledData = idfModel.transform(featurizedData)

            # split data into train and test
            training, test = spDF.randomSplit([0.8, 0.2], seed = 0)
            
            # naive bais classification
            nb = NaiveBayes(smoothing=1.0, labelCol='label', featuresCol='features')
            
            # configure ML pipeline
            pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, nb])
            
            # fit the pipeline to training documents
            print("Initiating training...")
            self.model = pipeline.fit(training)
            del training
            gc.collect()
            print("Completed training. Generating classification result...")
            
            # make predictions
            predictions = self.model.transform(test)

            # evaluate prediction
            TP = predictions.select("label", "prediction").filter((predictions.label == 1) & (predictions.prediction == 1)).count()
            TN = predictions.select("label", "prediction").filter((predictions.label == 0) & (predictions.prediction == 0)).count()
            FP = predictions.select("label", "prediction").filter((predictions.label == 0) & (predictions.prediction == 1)).count()
            FN = predictions.select("label", "prediction").filter((predictions.label == 1) & (predictions.prediction == 0)).count()
            total = predictions.select("label").count()

            accuracy	= (TP + TN) / total
            precision   = TP / (TP + FP)
            recall      = TP / (TP + FN)
            F1		= 2/(1/precision + 1/recall)

            print('accuracy:', accuracy)
            print('precision:', precision)
            print('recall:', recall)
            print('F1:', F1)
          
            del test
            gc.collect()

            print("Storing model info into disk...")
            self.model.save(constant.MODEL_DIR)

        else:
            print("Found existing model stored in the file. Loading...")
            self.model = PipelineModel.load(constant.MODEL_DIR)

        # to measure programing execution time
        print("--- time spent for initializing the classifier : %s seconds ---" % (time.time() - start_time))

        
    def stopClassifier(self):
        # close spark session
        self.sc.stop()

    # tokenizer only works off of dataframe, need to convert text to dataframe
    # before we can pass it to the stored ML pipeline
    def txtIsToxic(self, txt):
        df = self.sqlContext.createDataFrame([txt], StringType())
        txtDF = df.withColumnRenamed('value', 'comment_text')
   
        return self.isToxic(txtDF)
    
    # with given parameter s, it returns whether s is toxic or not
    # it is not expecting any arrays, it should be just single string value
    def isToxic(self, s):

        pred_df = self.model.transform(s)
        pred_list = pred_df.select('prediction').collect()
        return np.any(pred_list)
        

# main function if you need to run it separated, not through chatbot.py.
# The function will load local test CSV file and execute the prediction, instead of getting messages from TwitchTV channel it has deployed.
# With "test/test.csv", the code is going to take anothe 300MB +- to load all the text data and process it.
def main():
    print("Initiating...")

    # below file has smaller set of test data. Enable it instead if you want quicker testing
    test_data_path = "data/test_sample.csv"
    # below file has full set of test data. Try with it if you see more dresult. Beware : it will take some time.

    toxicClassifier = ToxicityClassifier()

    # loading chat logs from csv file
    df = toxicClassifier.sqlContext.read.csv(test_data_path, 
                                       header="true", 
                                       multiLine=True, 
                                       inferSchema=True,
                                       escape="\"")

    # transform test data's chat log to the existing vecorizer so it can be used for prediction        
    preds_list = toxicClassifier.model.transform(df).select('prediction').collect()
    preds = [row.prediction for row in preds_list]
    text = df.select('comment_text').collect()
    #print('preds', preds, 'text', text)
    for i, p in enumerate(preds):
        if p == 1:
            print("TOXIC>>> " + text[i].comment_text)            
            
    toxicClassifier.stopClassifier()
# just in case if need to run the test locally without TwitchBox working together
if __name__ == "__main__":

    if not sys.version_info[:1] == (3,):
        print(sys.version_info[:1] )
        sys.stderr.write("Python version 3 is required.\n")
        exit(1)

    main()
