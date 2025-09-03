from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout

def build_lstm(input_shape, units=50):
    """Builds an LSTM model."""
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=units),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_rnn(input_shape, units=50):
    """Builds a Simple RNN model."""
    model = Sequential([
        SimpleRNN(units=units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        SimpleRNN(units=units),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_gru(input_shape, units=50):
    """Builds a GRU model."""
    model = Sequential([
        GRU(units=units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(units=units),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model
