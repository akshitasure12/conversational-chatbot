import spacy
import numpy as np
nlp = spacy.load("en_core_web_sm")

def tokenize(sentence):
    doc = nlp(sentence)
    return [token.text for token in doc]

def lemmatize(word):
    doc = nlp(word)
    return doc[0].lemma_

def bag_of_words(tokenized_sentence, all_words):
    lemmatized_words = [lemmatize(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    for idx, w in enumerate(all_words):
        if w in lemmatized_words:
            bag[idx] = 1.0
            
    return bag
    
    