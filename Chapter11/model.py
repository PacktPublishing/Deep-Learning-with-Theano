from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense

def build_networks(input_shape, output_shape):
    state = Input(shape=input_shape)
    h = Conv2D(16, (8, 8) , strides=(4, 4), activation='relu', data_format="channels_first")(state)
    h = Conv2D(32, (4, 4) , strides=(2, 2), activation='relu', data_format="channels_first")(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu')(h)

    value = Dense(1, activation='linear', name='value')(h)
    policy = Dense(output_shape, activation='softmax', name='policy')(h)

    value_network = Model(inputs=state, outputs=value)
    policy_network = Model(inputs=state, outputs=policy)
    train_network = Model(inputs=state, outputs=[value, policy])

    return value_network, policy_network, train_network
