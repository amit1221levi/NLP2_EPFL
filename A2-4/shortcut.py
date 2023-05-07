import itertools
import jsonlines
import nltk
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
stop_words.append('uh')

import string
puncs = string.punctuation

def word_pair_extraction(prediction_files, tokenizer):
    '''
    Extract all word pairs (word_from_premise, word_from_hypothesis) from input as features.
    
    INPUT: 
      - prediction_files: file path for all predictions
      - tokenizer: tokenizer used for tokenization
    
    OUTPUT: 
      - word_pairs: a dict of all word pairs as keys, and label frequency of values. 
    '''
    word_pairs = {}
    label_to_id = {"entailment": 0, "neutral": 1, "contradiction": 2}
    
    for pred_file in prediction_files:
        with jsonlines.open(pred_file, "r") as reader:
            for pred in reader.iter():
                #########################################################
                #          TODO: construct word_pairs dictionary        # 
                #  - tokenize the text with 'tokenizer'                 # 
                #  - pair words as keys (you can use itertools)         #
                #  - count predictions for each paired words as values  # 
                #  - remenber to filter undesired word pairs            # 
                #########################################################
                premise = pred["premise"]
                hypothesis = pred["hypothesis"]
                label = pred["prediction"]
                
                premise_tokens = tokenizer.tokenize(premise)
                hypothesis_tokens = tokenizer.tokenize(hypothesis)
                
                # Remove stopwords and punctuations from tokens
                premise_tokens = [token.lower() for token in premise_tokens if token.lower() not in stop_words and token not in puncs]
                hypothesis_tokens = [token.lower() for token in hypothesis_tokens if token.lower() not in stop_words and token not in puncs]
                
                for pair in itertools.product(premise_tokens, hypothesis_tokens):
                    if pair[0] != pair[1]:
                        key = (pair[0], pair[1])
                        label = label_to_id[pred["prediction"]]
                        if key not in word_pairs:
                            word_pairs[key] = [0, 0, 0]
                        word_pairs[key][label] += 1
                #####################################################
                #                   END OF YOUR CODE                #
                #####################################################
    
    return word_pairs
