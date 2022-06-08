import os

import numpy as np

import re

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

## https://pyenchant.github.io/pyenchant/
#import enchant

class DataPreprocess(object):
    def __init__(self: object):
        self.stop_words = None
        self.char_filter = None
        self.english_dict = None
    
    def get_words(self: object, text: str) -> str:
        # iterate over results of word_tokenize(...) and apply
        #  custom lambda function to lower case each word
        words = map(lambda word: word.lower(), word_tokenize(text))

        # filter out stop words from tokenize words from text
        #  parameter
        if not self.stop_words is None:
            words = [word for word in words if word not in self.stop_words]
        else:
            words = list(words)

        # filter out tokens containing non-alpha characters
        if not self.char_filter is None:
            words = list(
                filter(
                    lambda word: self.char_filter.match(word), words
                )
            )

        return ' '.join(words)
    
    def get_stems(self: object, text: str) -> str:
        # iterate over results of word_tokenize(...) and apply
        #  custom lambda function to lower case each word
        words = map(lambda word: word.lower(), word_tokenize(text))

        # use popular PorterStemmer to strip morphological word affixes
        #  and find word stems
        stems = (
            list(
                map(lambda word: PorterStemmer().stem(word), words)
            )
        )

        # filter out tokens containing non-alpha characters
        if not self.char_filter is None:
            stems = list(
                filter(
                    lambda stem : self.char_filter.match(stem), stems
                )
            )

        return ' '.join(stems)
    
    def get_lemmas(self: object, text: str) -> str:
        # iterate over results of word_tokenize(...) and apply
        #  custom lambda function to lower case each word
        words = map(lambda word: word.lower(), word_tokenize(text))

        # use popular WordNetLemmatizer to reduce word to it's base form
        lemmas = (
            list(
                map(lambda word: WordNetLemmatizer().lemmatize(word), words)
            )
        )

        # filter out tokens containing non-alpha characters
        if not self.char_filter is None:
            lemmas = list(
                filter(
                    lambda lemma : self.char_filter.match(lemma), lemmas
                )
            )

        return ' '.join(lemmas)
    
    def get_word_counts(self: object, text: str, cap: int = None) -> int:
        count = len(str(text).split())
        
        if not cap is None:
            cap = np.min([count, cap])
            
        return count
    
    def get_stop_words(self: object) -> list:
        if self.stop_words is None:
            self.stop_words = stopwords.words('english')
            
        return self.stop_words
    
    def get_char_filter(self: object) -> object:
        if self.char_filter is None:
            self.char_filter = re.compile('[a-zA-Z]+')
            
        return self.char_filter
    
    def get_english_dictionary_words(self: object) -> list:
        if self.english_dict is None:
            # set to None to exclude checking if words exist in an English dictionary
            self.english_dict = None #enchant.Dict("en_US")
            
        return self.english_dict
    
    def preprocess_data(self: object, target_df: object, max_sequence_length:int = None) -> object:
        _ = self.get_stop_words()
        _ = self.get_char_filter()
        
        target_df['tweet_adj'] = target_df['tweet'].apply(self.get_words)
        
        target_df['stems'] = target_df['tweet_adj'].apply(self.get_stems)
        
        target_df['lemmas'] = target_df['tweet_adj'].apply(self.get_lemmas)
        
        target_df['lengths'] = target_df['tweet_adj'].apply(self.get_word_counts)
        
        if not max_sequence_length is None:
            target_df['lengths_capped'] = target_df['lengths'].apply(lambda t : np.min([t, max_sequence_length]))
        
        target_df['target'] = 0
        target_df.loc[target_df['label'] == 'fake', 'target'] = 1
        target_df.loc[target_df['label'] == 'real', 'target'] = 0
        
        return target_df
    
    def basic_tokenize(self: object, text: str, calc_stems = False, calc_lemmas = False, stop_words: list = [], char_filter : re.Pattern = None) -> tuple:
        # iterate over results of word_tokenize(...) and apply
        #  custom lambda function to lower case each word
        words = map(lambda word: word.lower(), word_tokenize(text))

        # filter out stop words from tokenize words from text
        #  parameter
        words = [word for word in words if word not in stop_words]

        stems = []
        lemmas = []
        
        if calc_stems:
            # use popular PorterStemmer to strip morphological word affixes
            #  and find word stems
            stems = (
                list(
                    map(lambda word: PorterStemmer().stem(word), words)
                )
            )

        if calc_lemmas:
            # use popular WordNetLemmatizer to reduce word to it's base form
            lemmas = (
                list(
                    map(lambda word: WordNetLemmatizer().lemmatize(word), words)
                )
            )

        # filter out tokens containing non-alpha characters
        if not char_filter is None:
            words = list(
                filter(
                    lambda word: char_filter.match(word), words
                )
            )
            
            if calc_stems:
                stems = list(
                    filter(
                        lambda stem : char_filter.match(stem), stems
                    )
                )

            if calc_lemmas:
                lemmas = list(
                    filter(
                        lambda lemma : char_filter.match(lemma), lemmas
                    )
                )

        return words, stems, lemmas
    
    def _custom_tokenizer(self: object, text: str) -> list:
        words, stems, lemmas = self.basic_tokenize(text, calc_stems = True, calc_lemmas = True, char_filter = self.char_filter, stop_words = self.stop_words)

        return words
    
    def get_doc_term_matrix(self: object, records_list: list, ngram_range: tuple = (1, 3), min_max_range: tuple = (.10, .60)) -> tuple:
        '''
            min_max_range: # 10% < docs < 60%
        '''
        _ = self.get_stop_words()
        _ = self.get_char_filter()
        
        vectorizer = TfidfVectorizer(
            tokenizer = self._custom_tokenizer,
            norm='l2',
            ngram_range = ngram_range,
            min_df = min_max_range[0],
            max_df = min_max_range[1]
        )
        doc_term_matrix = vectorizer.fit_transform(records_list)
        
        #get_feature_names()
        return vectorizer.get_feature_names_out(), doc_term_matrix, vectorizer.vocabulary_, vectorizer
    
    
# ------------------------------------------------------
# CLASS TO LOAD AND PREPARE PRETRAINED EMBEDDINGS
# ------------------------------------------------------
class PretrainedEmbeddingsInfo(object):
    def __init__(self, parent_data_dir:str = ''):
        # -----------------------------------------------------------------
        # GLOVE DETAILS
        # -----------------------------------------------------------------
        #  https://nlp.stanford.edu/pubs/glove.pdf
        #  https://nlp.stanford.edu/projects/glove/
        #  "tokens": refers to the total number of "words" in a corpus
        #  "vocabulary": refers to the total number of "unique word"
        # -----------------------------------------------------------------
        self.glove_wiki_folder_path = 'glove'
        
        if not parent_data_dir is None and len(parent_data_dir) > 0:
            self.glove_wiki_folder_path = os.path.join(parent_data_dir, self.glove_wiki_folder_path)
        
        self.glove_wiki_file_format    = 'glove.6B.{}d.txt'
        self.glove_wiki_emb_dimensions = [50, 100, 200, 300]
        
        self.vocab      = None
        self.embeddings = None
        
        self.file_path = ""
    
    # -----------------------
    # PROPERTIES
    # -----------------------
    @property
    def Vocab(self):
        return self.vocab
    
    @property
    def Embeddings(self):
        return self.embeddings
        
    # -----------------------
    # PUBLIC METHODS
    # -----------------------
    def load(self, embedding_dim, special_tokens):
        if not embedding_dim in self.glove_wiki_emb_dimensions:
            raise Exception('Unsupported dimension size for GloVe Wiki pre-trained embeddings')

        self.file_path = os.path.join(self.glove_wiki_folder_path, self.glove_wiki_file_format.format(embedding_dim))
        self.vocab = self.__load_glove_common_tokens(self.file_path, special_tokens = special_tokens)
        self.embeddings = self.__get_glove_embeddings(embedding_dim, self.file_path, special_tokens = special_tokens)
    
    # -----------------------
    # PRIVATE METHODS
    # -----------------------
    def __load_glove_common_tokens(self, file_path, special_tokens = None):
        vocab_dict = {}
        
        j = 0
        
        if not special_tokens is None:
            for special_token in special_tokens:
                vocab_dict[special_token] = j
                j += 1
        
        for line in open(file_path):
            line = line.strip() # remove leading and trailing whitespaces
            word = line.split(' ')[0]
            vocab_dict[word] = j
            j += 1
        
        return vocab_dict
    
    def __get_glove_embeddings(self, dim, file_path, special_tokens = None):
        j = 0
        embeddings = np.zeros([len(self.vocab), dim])
        
        if not special_tokens is None:
            unk_embeddings = self.__generate_unknown_glove_vector(file_path)
            
            for special_token in special_tokens:
                embeddings[j] = unk_embeddings
                j += 1
        
        for line in open(file_path):
            line = line.strip() # remove leading and trailing whitespaces
            line = line.split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in self.vocab:
                word_index = j
                j += 1
                embeddings[word_index] = np.asarray(embedding)
                
        return embeddings
    
    def __generate_unknown_glove_vector(self, file_path):
        # Jeffrey Pennington (GloVe author):
        #  (1) The pre-trained vectors do not have an unknown token, and currently the code just
        #      ignores out-of-vocabulary words when producing the co-occurrence counts
        #  (2) I've found that just taking an average of all or a subset of the word
        #      vectors produces a good unknown vector
        # URL => https://stackoverflow.com/questions/49239941/what-is-unk-in-the-pretrained-glove-vector-files-e-g-glove-6b-50d-txt
        
        # Get number of vectors and hidden dim
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                pass
        n_vec = i + 1
        hidden_dim = len(line.split(' ')) - 1

        vecs = np.zeros((n_vec, hidden_dim), dtype=np.float32)

        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                vecs[i] = np.array([float(n) for n in line.split(' ')[1:]], dtype=np.float32)

        average_vec = np.mean(vecs, axis=0)
        
        return average_vec
