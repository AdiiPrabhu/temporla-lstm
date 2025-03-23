import zipfile
import os
import shutil

ZIP_PATH = 'E:/daicwoz.zip'
DEST_DIR = 'E:/daicwoz'

# Define output folders
transcripts_dir = os.path.join(DEST_DIR, 'transcripts')
audio_dir = os.path.join(DEST_DIR, 'audio')

# Create directories if they don't exist
os.makedirs(transcripts_dir, exist_ok=True)
os.makedirs(audio_dir, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    file_list = zip_ref.namelist()

    for file in file_list:
        # Copy transcripts
        if file.endswith('_TRANSCRIPT.csv'):
            filename = os.path.basename(file)
            dest_path = os.path.join(transcripts_dir, filename)
            with zip_ref.open(file) as source, open(dest_path, 'wb') as target:
                shutil.copyfileobj(source, target)
            print(f'Transcript copied: {filename}')

        # Copy audio files
        elif file.endswith('_AUDIO.wav'):
            filename = os.path.basename(file)
            dest_path = os.path.join(audio_dir, filename)
            with zip_ref.open(file) as source, open(dest_path, 'wb') as target:
                shutil.copyfileobj(source, target)
            print(f'Audio copied: {filename}')

        # Copy labels file
        elif file.endswith('train_split_Depression_AVEC2017.csv'):
            labels_dest = os.path.join(DEST_DIR, 'labels.csv')
            with zip_ref.open(file) as source, open(labels_dest, 'wb') as target:
                shutil.copyfileobj(source, target)
            print(f'Labels file copied: labels.csv')

print('Dataset successfully organized!')
