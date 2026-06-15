# rebuild_lstm.py
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.models import Model

# Paths
old_model_path = r"C:\Users\dedee\Downloads\epidemic-project\models\global_lstm.h5"
new_model_path = r"C:\Users\dedee\Downloads\epidemic-project\models\global_lstm.keras"

# Rebuild the model architecture (same as your original code)
seq_len = 14
n_features = 5
n_regions = 46
embed_dim = 8

seq_in = Input(shape=(seq_len, n_features), name='seq_in')
region_in = Input(shape=(1,), dtype='int32', name='region_in')

x = LSTM(64, activation='tanh', name='lstm_enc')(seq_in)
x = Dropout(0.2)(x)

emb = Embedding(input_dim=n_regions, output_dim=embed_dim, name='region_emb')(region_in)
embf = Flatten()(emb)

concat = Concatenate()([x, embf])
dense = Dense(64, activation='relu')(concat)
dense = Dropout(0.2)(dense)
out = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[seq_in, region_in], outputs=out)

# Load weights from old h5
model.load_weights(old_model_path)

# Save in Keras 3 format
model.save(new_model_path, save_format='keras')
print(f"Model successfully re-saved to {new_model_path}")
