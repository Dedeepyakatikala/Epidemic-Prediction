# re_save_lstm.py
import tensorflow as tf

# Path to your old h5 model
old_model_path = r"C:\Users\dedee\Downloads\epidemic-project\models\global_lstm.h5"

# Path to save the new model in .keras format
new_model_path = r"C:\Users\dedee\Downloads\epidemic-project\models\global_lstm.keras"

# Load the old model (ignore compile if needed)
model = tf.keras.models.load_model(old_model_path, compile=False)

# Optionally, you can compile it again (if needed for training)
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Save in new TF/Keras 3 compatible format
model.save(new_model_path, save_format='keras')

print(f"Model successfully re-saved to {new_model_path}")
