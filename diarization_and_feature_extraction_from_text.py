import assemblyai as aai
import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
#nltk.download('punkt_tab')


text_features = pd.DataFrame(columns=['filename','Word Variance','Hapax Legomena'])
# Replace with your API key
aai.settings.api_key = "262ec24608c0442483768e3004d70d53"
config = aai.TranscriptionConfig(speaker_labels=True)
FILE_URL="/home/droidis/Desktop/train_edited"
directory = os.fsencode(FILE_URL)



for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".mp3"):
      url=FILE_URL+ '/' + filename
      transcriber = aai.Transcriber()
      transcript = transcriber.transcribe(
        url,
        config=config
      )

      for utterance in transcript.utterances:
        print(f"Speaker {utterance.speaker}: {utterance.text}")
        text = utterance.text

    tokens = word_tokenize(text.lower())

    # Frequency distribution
    fdist = FreqDist(tokens)
    word_variance= len(fdist) / len(tokens) # Lexical richnes
    hapax_legomena = len([word for word, freq in fdist.items() if freq == 1])
    text_features=text_features._append({'filename':filename, 'Word Variance':word_variance, 'Hapax Legomena':hapax_legomena}, ignore_index=True)

text_features=text_features.sort_values('filename')
print(text_features)