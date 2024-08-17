import numpy as np
import nltk
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop_words
from spacy.lang.en.stop_words import STOP_WORDS as en_stop_words
from spacy.tokens import Doc



from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def tokenize(sentence):
    if isinstance(sentence, bytes):
        sentence = sentence.decode('utf-8')
    return nltk.word_tokenize(sentence)



def stem(word):
    if isinstance(word, bytes):
        word = word.decode('utf-8')
    return stemmer.stem(word.lower())



def bag_of_words(tokenized_sentence, words):
   
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag