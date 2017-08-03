# Chapter 11: Learning from the Environment with Reinforcement

Get the data:

    pip install gym
    pip install gym[atari]

Install the requirements:

    conda install --file requirements.txt

Test the OpenGym API:

    python 0-gym-api.py

Train the model on 16 cores:

    python 1-train.py --game=Breakout-v0 --processes=16

Play the game with the learned model:

    python 2-play.py --game=Breakout-v0 --model=model-Breakout-v0-35750000.h5
