import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import pandas as pd
import assemblyai as aai


def feature_extraction_from_text(audios):
# Download NLTK tokenizer if missing
    nltk.download('punkt')
    print("Diarizing and extracting features from text...")

    # ✅ Creating DataFrame just like before
    text_features = pd.DataFrame(columns=['filename', 'Word Variance', 'Hapax Legomena'])

    # ✅ Replace with your API key
    aai.settings.api_key = "262ec24608c0442483768e3004d70d53"
    config = aai.TranscriptionConfig(speaker_labels=True)

    directory = os.fsencode(audios)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        if filename.endswith(".mp3"):
            url = os.path.join(audios, filename)  # Ensure correct file path
            transcriber = aai.Transcriber()

            try:
                transcript = transcriber.transcribe(url, config=config)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue  # Skip file if transcription fails

            # ✅ Combine all speakers' utterances into one text block per audio file
            full_text = " ".join([utterance.text for utterance in transcript.utterances])

            if not full_text.strip():
                print(f"No speech detected in {filename}, skipping...")
                continue  # Skip empty transcriptions

            tokens = word_tokenize(full_text.lower())  # Tokenize text

            if not tokens:
                print(f"No valid words in {filename}, skipping...")
                continue  # Skip if no tokens

            # ✅ Compute features on the combined text
            fdist = FreqDist(tokens)
            word_variance = len(fdist) / len(tokens)  # Lexical richness
            hapax_legomena = sum(1 for freq in fdist.values() if freq == 1)  # Count words appearing only once

            # ✅ Append row to DataFrame just like before
            text_features = text_features._append(
                {'filename': filename, 'Word Variance': word_variance, 'Hapax Legomena': hapax_legomena},
                ignore_index=True
            )

    # ✅ Sorting and saving to CSV just like before
    text_features = text_features.sort_values('filename')
    text_features.to_csv('text_features.csv', index=False)