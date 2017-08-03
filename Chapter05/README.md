# Chapter 5: Analyzing Sentiment with a bidirectional LSTM

Install Keras:

    conda install keras

Download the data:

    wget http://alt.qcri.org/semeval2014/task9/data/uploads/semeval2013_task2_train.zip -P /sharedfiles
    wget http://alt.qcri.org/semeval2014/task9/data/uploads/semeval2013_task2_dev.zip -P /sharedfiles
    wget http://alt.qcri.org/semeval2014/task9/data/uploads/semeval2013_task2_test_fixed.zip  -P /sharedfiles
    unzip /sharedfiles/semeval2013_task2_train.zip
    unzip /sharedfiles/semeval2013_task2_dev.zip
    unzip /sharedfiles/semeval2013_task2_test_fixed.zip

Install BS4:

    pip install bs4

Convert the data:

    python download_tweets.py train/cleansed/twitter-train-cleansed-A.tsv > sem_eval2103.train
    python download_tweets.py dev/gold/twitter-dev-gold-A.tsv > sem_eval2103.dev
    python download_tweets.py SemEval2013_task2_test_fixed/gold/twitter-test-gold-A.tsv > sem_eval2103.test

Train the model:

    python bilstm.py
