'''
v1: two dense 130
'''
from keras.models import Sequential
from keras.layers import Dense


# define the model
def get_model_keras():
    # create model
    model = Sequential()
    model.add(Dense(130, kernel_initializer='normal', activation='relu'))
    model.add(Dense(130, kernel_initializer='normal'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
