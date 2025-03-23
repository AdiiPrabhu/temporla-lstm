import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

DATA_PATH = 'E:/daicwoz'
TRANSCRIPTS_PATH = os.path.join(DATA_PATH, 'transcripts')
LABELS_PATH = os.path.join(DATA_PATH, 'labels.csv')

# Load labels
labels_df = pd.read_csv(LABELS_PATH)
labels_df = labels_df[['Participant_ID', 'PHQ8_Binary']]  # labels: 0 or 1

# CORRECTED FUNCTION: Load transcripts with tab separator clearly defined
def load_transcript(participant_id):
    transcript_file = os.path.join(TRANSCRIPTS_PATH, f"{participant_id}_TRANSCRIPT.csv")
    if os.path.exists(transcript_file):
        # Specify tab delimiter
        df = pd.read_csv(transcript_file, delimiter='\t')
        transcript = ' '.join(df['value'].dropna().astype(str))
        return transcript
    else:
        return None

# Load transcripts
labels_df['transcript'] = labels_df['Participant_ID'].apply(load_transcript)

# Drop missing transcripts
labels_df.dropna(subset=['transcript'], inplace=True)

print(f"Loaded {len(labels_df)} transcripts successfully.")

# Tokenization parameters
MAX_WORDS = 10000
MAX_SEQ_LEN = 200  

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(labels_df['transcript'])

sequences = tokenizer.texts_to_sequences(labels_df['transcript'])

X = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
y = labels_df['PHQ8_Binary'].values

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, {y_train.shape}")
print(f"Test shape: {X_test.shape}, {y_test.shape}")

# Save tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Preprocessing and feature extraction completed successfully.")
