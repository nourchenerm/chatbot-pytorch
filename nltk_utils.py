import numpy as np
import nltk
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop_words
from spacy.lang.en.stop_words import STOP_WORDS as en_stop_words
from spacy.tokens import Doc
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
nlp_fr = spacy.load('fr_core_news_md')
nlp_en = spacy.load('en_core_web_md')
def tokenize(sentence):
    
    return nltk.word_tokenize(sentence)


def stem(word):
   
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