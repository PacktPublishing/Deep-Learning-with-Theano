# Chapter 10: predicting times sequences with Advanced RNN

Train a stack of 3 simple RNN:

    THEANO_FLAGS="device=cuda1,optimizer_excluding=scanOp_pushout_output" python train.py --model rnn --num_layers 3 --learning_rate 0.02

Train a stack of 3 simple LSTM:

    THEANO_FLAGS="device=cuda1,optimizer_excluding=scanOp_pushout_output" python train.py --model lstm --num_layers 3

Train one layer of RHN:

    THEANO_FLAGS="device=cuda1,optimizer_excluding=scanOp_pushout_output" python train.py  --model rhn --depth 10 --num_layers 1

For the last model, you should get a perplexity of 64.
