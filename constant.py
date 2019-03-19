# filename that will be used to save and restore the trained classifier model to/from the file.
CLASSIFIER_FILE = "classifier.joblib"

# filename that will be used to save and restore the fitted vectorizer to/from the file.
VECTORIZER_FILE = "vectorizer.joblib"

# how # of lines that panda's read_csv is going to read per each time. 
# It's not byte size or filesize to read. 
# REDUCE THIS NUMBER IF YOUR PYTHON PROCESS IS GETTING KILLED WHILE RUNNING. It's pretty much due to code taking too much memory for processing.
# With 15,000, it's expecting to have about 6GB of memory maximum in Mac OS (compressed size less than 1GB) with current training data and takes about 3~4 minutes to complete the training.
# Runing time can vary depends on the system.
CVS_CHUNKSIZE = 15000




# has first 161060 sample data(lines)
TRAINING_DATA_PATH = "data/train_sample_161060.csv"

# has first 8241 sample data(lines)
# TRAINING_DATA_PATH = "data/train_sample_8241.csv" 

# Following line's datafile is huge and python process will be killed while it is running.. or maybe taking forever with partial_fit
# there are two other training sample data with project, which is data/train_sample_161060.csv and data/train_sample_8241.csv
# that suffix means the line of traning data in the file. Change it to smaller one to see how Detox behaves with different training size/set.
# TRAINING_DATA_PATH = "data/train.csv"
