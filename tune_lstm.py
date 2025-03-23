import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split

# Load preprocessed data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

VOCAB_SIZE = 10000
MAX_SEQ_LEN = 200

# Define model building function
def build_model(hp):
    model = Sequential()
    
    # Embedding layer clearly defined
    model.add(Embedding(VOCAB_SIZE,
                        hp.Choice('embedding_dim', [64, 128, 256]),
                        input_length=MAX_SEQ_LEN))
    
    # LSTM or BiLSTM
    if hp.Choice('bidirectional', [True, False]):
        model.add(Bidirectional(LSTM(
            hp.Choice('lstm_units', [32, 64, 128]), return_sequences=True)))
        model.add(Dropout(hp.Choice('dropout', [0.2, 0.3, 0.5])))
        model.add(Bidirectional(LSTM(hp.Choice('lstm_units2', [32, 64]))))
    else:
        model.add(LSTM(
            hp.Choice('lstm_units', [32, 64, 128]), return_sequences=True))
        model.add(Dropout(hp.Choice('dropout', [0.2, 0.3, 0.5])))
        model.add(LSTM(hp.Choice('lstm_units2', [32, 64])))

    # Dense layers
    model.add(Dense(hp.Choice('dense_units', [32, 64]), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model clearly
    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# Keras Tuner setup clearly
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10, # adjust clearly for more experiments
    executions_per_trial=2, # stable results
    directory='lstm_tuning',
    project_name='depression_detection'
)

# Tuner summary clearly shown
tuner.search_space_summary()

# Run hyperparameter tuning clearly
tuner.search(X_train, y_train,
             epochs=10,
             batch_size=16,
             validation_split=0.2,
             verbose=1)

# Get the optimal hyperparameters clearly
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("The best hyperparameters clearly found:")
print(f"Embedding Dimension: {best_hps.get('embedding_dim')}")
print(f"Bidirectional: {best_hps.get('bidirectional')}")
print(f"LSTM Units (Layer 1): {best_hps.get('lstm_units')}")
print(f"LSTM Units (Layer 2): {best_hps.get('lstm_units2')}")
print(f"Dropout Rate: {best_hps.get('dropout')}")
print(f"Dense Layer Units: {best_hps.get('dense_units')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")

# Train the best model clearly
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=16,
                    validation_split=0.2,
                    verbose=1)

# Save the clearly trained best model
model.save('best_tuned_lstm.h5')

print("Best model clearly trained and saved successfully.")
