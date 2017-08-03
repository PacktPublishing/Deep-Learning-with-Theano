# Chapter 13 : extending Deep Learning with Theano

Run your own new AXPB operator written in Python for the CPU:

    python 1-python-op.py

PyGPU library enables to use the GPU directly from Python code:

    python 2-pygpu-example.py

Run your own new AXPB operator written in Python for the GPU thanks to PyGpu library:

    python 3-python-gpu-op.py

Run your own new AXPB operator written in C for the CPU:

    python 4-C-cpu-op.py

Note that to run operators written with GPU Kernels, you need to downgrade PyGpu and libgpuarray libraries:

    conda install libgpuarray=0.6.4-0 pygpu=0.6.4
    python 5a-C-gpu-op-from-theano.py

Run your own new operator written in C for the GPU:

    python 5-C-gpu-op.py

For the Google ML Cloud example, create a bucket and upload data:

    gsutil mb -l europe-west1 gs://keras_sentiment_analysis
    gsutil cp -r 7-google-cloud/data/sem_eval2103.train gs://keras_sentiment_analysis/sem_eval2103.train
    gsutil cp -r 7-google-cloud/data/sem_eval2103.dev gs://keras_sentiment_analysis/sem_eval2103.dev
    gsutil cp -r 7-google-cloud/data/sem_eval2103.test gs://keras_sentiment_analysis/sem_eval2103.test

Check the code runs locally:

    gcloud ml-engine local train --module-name 7-google-cloud.bilstm \
      --package-path ./7-google-cloud  -- --job-dir ./7-google-cloud \
      -t 7-google-cloud/data/sem_eval2103.train \
      -d 7-google-cloud/data/sem_eval2103.dev \
      -v 7-google-cloud/data/sem_eval2103.test

If everything works fine locally, to submit it to the cloud:

    JOB_NAME="keras_sentiment_analysis_train_$(date +%Y%m%d_%H%M%S)"

    gcloud ml-engine jobs submit training $JOB_NAME \
              --job-dir gs://keras_sentiment_analysis/$JOB_NAME \
              --runtime-version 1.0 \
              --module-name 7-google-cloud.bilstm  \
              --package-path ./7-google-cloud \
              --region europe-west1 \
              --config=7-google-cloud/cloudml-gpu.yaml \
              -- \
              -t gs://keras_sentiment_analysis/sem_eval2103.train \
              -d gs://keras_sentiment_analysis/sem_eval2103.dev \
              -v gs://keras_sentiment_analysis/sem_eval2103.test

    gcloud ml-engine jobs describe $JOB_NAME
