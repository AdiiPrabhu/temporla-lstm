import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Paths
DATA_PATH = 'E:\daicwoz'
TRANSCRIPTS_PATH = os.path.join(DATA_PATH, 'transcripts')
LABELS_PATH = os.path.join(DATA_PATH, 'labels.csv')

# Load labels
labels_df = pd.read_csv(LABELS_PATH)
labels_df = labels_df[['Participant_ID', 'PHQ8_Binary']]

# Load transcripts (tab-separated files)
def load_transcript(participant_id):
    transcript_file = os.path.join(TRANSCRIPTS_PATH, f"{participant_id}_TRANSCRIPT.csv")
    if os.path.exists(transcript_file):
        df = pd.read_csv(transcript_file, delimiter='\t')
        transcript = ' '.join(df['value'].dropna().astype(str))
        return transcript
    else:
        return None

labels_df['transcript'] = labels_df['Participant_ID'].apply(load_transcript)

# Drop rows with no transcripts
labels_df.dropna(subset=['transcript'], inplace=True)

# Tokenization parameters
MAX_WORDS = 10000
MAX_SEQ_LEN = 200

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(labels_df['transcript'])

# Text to sequences
sequences = tokenizer.texts_to_sequences(labels_df['transcript'])

# Padding
X = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
y = labels_df['PHQ8_Binary'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save processed data
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# Save tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Preprocessing completed and data saved successfully.")
