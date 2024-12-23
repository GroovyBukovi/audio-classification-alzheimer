import spacy

from spacy.lang.en.examples import sentences
# Load spaCy model

nlp = spacy.load("en_core_web_sm")

# Process text
doc = nlp("Well, the kids are working in the corner, grading. They are going to get some cookies from the cookie jar. And the mother does not see it because she's inside drying the clothes. And the kids then just, I guess, picture that mother was working hard and the kids were playing, and all of a sudden, somebody turned over a dish all over the floor. Except that it did not dry. It didn't splash from the it splashed from the sink, but not from the no. Trying to get too much out of one of the kids. Going to get a crack on the head. And maybe has a Sometimes I see very clear. Other times I see a weak image. Sometimes I just have this what is it? Mostly I have not so much trouble of looking at a thing as an image. But not but not getting any.")

# Analyze word variance
unique_words = set([token.text.lower() for token in doc if token.is_alpha])
lexical_richness = len(unique_words) / len([token for token in doc if token.is_alpha])
print("Lexical Richness:", lexical_richness)

import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
#nltk.download('punkt_tab')

# Sample text
text = "Well, the kids are working in the corner, grading. They are going to get some cookies from the cookie jar. And the mother does not see it because she's inside drying the clothes. And the kids then just, I guess, picture that mother was working hard and the kids were playing, and all of a sudden, somebody turned over a dish all over the floor. Except that it did not dry. It didn't splash from the it splashed from the sink, but not from the no. Trying to get too much out of one of the kids. Going to get a crack on the head. And maybe has a Sometimes I see very clear. Other times I see a weak image. Sometimes I just have this what is it? Mostly I have not so much trouble of looking at a thing as an image. But not but not getting any."
text2 = "Hello world"
# Tokenize text
tokens = word_tokenize(text.lower())

# Frequency distribution
fdist = FreqDist(tokens)
print("Word Variance:", len(fdist) / len(tokens))  # Lexical richness
print("Hapax Legomena:", len([word for word, freq in fdist.items() if freq == 1]))





