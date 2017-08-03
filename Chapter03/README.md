# Chapter 3: Encoding words into vectors

Get the data:

    wget http://mattmahoney.net/dc/text8.zip -O /sharedfiles/text8.gz
    gzip -d /sharedfiles/text8.gz -f

Install NLTK lib:

    conda install nltk

Download the english nltk data:

    python -c "import nltk; nltk.download('book')"

Train CBOW model:

    python 1-train-CBOW.py

Plot:

    python 2-plot.py

Evaluate:

    python 3-evaluate.py
