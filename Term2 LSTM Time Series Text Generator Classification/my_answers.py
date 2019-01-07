import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# The function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series:list, window_size:int, futureLen:int=1):
    # containers for input/output pairs
    X, y = [], []
    series = np.array(series)                   # Convert numpy Array
    # Get Targets y
    y = series[window_size + futureLen - 1:]    # y Targets; -1: for inclusive start
    y = np.reshape(y, (len(y), 1))
    # Loop to get Sliding Window'd Series
    xi = series[:-futureLen]                    # X remove future values that have no corrisponding y
    for end in range(window_size, len(xi)+1):   
        X.append(xi[end - window_size :end])    # Window Cascaded start; xi[start:end]
    return np.array(X),y

# RNN Model to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))    # 1st Layer: LSTM
    model.add(Dense(1, activation=None))                # 2nd Layer: Dense
    model.summary()
    return model


### Returns the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    from collections import OrderedDict
    #
    punctuation = ['!', ',', '.', ':', ';', '?']
    tx = text.lower()                                   # All text to lowercase
    uniqText = list(OrderedDict.fromkeys(tx).keys())    # Unique Set of Chars in Text
    uniqText.remove(' ')                                # Space is exempt
    for char in uniqText:
        if char.isalpha():                              # If Letter do nothing, continue
            try: 
                if char.encode('ascii').isalpha(): continue    # ascii
            except: pass
        elif char in punctuation: continue              # if excepted punc. do nothing, continue
        tx = tx.replace(char, ' ')                      # replace atypical chars. with a space
    return tx

### Function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    # Loop by Sliding Window of step_size
    for i in range(window_size, len(text), step_size):
        inputs.append(text[i-window_size:i])            # Windowed Inputs back
        outputs.append(text[i])                         # Future Char, i: not i+1 since it is not included in the slice
    return inputs,outputs

# RNN model structure according to Jupyter Notebook structure
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))      # input Layer: LSTM
    model.add(Dense(num_chars, activation='softmax'))               # Output Layer: Dense Softmax
    model.summary()
    return model


# if __name__ == '__main__':
#     # print(window_transform_text('abcdefghijklmnopqrstuvwxyz', 5, 3))
#     print(cleaned_text('\xe3^^aabbcc,. && ** ))'))
#     # window_transform_series(list(range(100)), 5)
#     print()