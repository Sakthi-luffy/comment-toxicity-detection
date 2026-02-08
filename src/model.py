from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def build_model(vocab_size, max_len, output_dim):
    model = Sequential([
        Embedding(vocab_size, 128),
        LSTM(64),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(output_dim, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
