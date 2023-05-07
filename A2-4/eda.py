import random
import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
  
from nltk.corpus import wordnet, stopwords

# ========================== Synonym Replacement ========================== #
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)

def synonym_replacement(sentence, n):
    
    words = sentence.split()
    
    ############################################################################
    # TODO: Replace up to n random words in the sentence with their synonyms.  #
    #   You should                                                             #
    #   - (i)   replace random words with one of its synonyms, until           #
    #           the number of replacement gets to n or all the words           #
    #           have been replaced;                                            #
    #   - (ii)  NO stopwords should be replaced!                               #
    #   - (iii) return a new sentence after all the replacement.               #
    ############################################################################
    # Replace "..." with your code
    new_sentence = []
    count = 1
    for word in words:
        if word not in stopwords.words('english') and random.random() < n/count:
            syns = get_synonyms(word)
            if syns:
                new_sentence.append(random.choice(syns))
            else:
                new_sentence.append(word)
        else:
            new_sentence.append(word)
        count += 1

    new_sentence =  ' '.join(new_sentence)
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################

    return new_sentence


# ========================== Random Deletion ========================== #
def random_deletion(sentence, p, max_deletion_n):

    words = sentence.split()
    max_deletion_n = min(max_deletion_n, len(words)-1)
    
    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    ############################################################################
    # TODO: Randomly delete words with probability p. You should               #
    # - (i)   iterate through all the words and determine whether each of them #
    #         should be deleted;                                               #
    # - (ii)  you can delete at most `max_deletion_n` words;                   #
    # - (iii) return the new sentence after deletion.                          #
    ############################################################################



    new_sentence = []
    for word in words:
        if random.random() > p and max_deletion_n > 0:
            max_deletion_n -= 1
        else:
            new_sentence.append(word)

    return ' '.join(new_sentence)

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    
    return new_sentence


# ========================== Random Swap ========================== #
def swap_word(sentence):
    
    words = sentence.split()
    if len(words) <= 1:
      return sentence
    ############################################################################
    # TODO: Randomly swap two words in the sentence. You should                #
    # - (i)   randomly get two indices;                                        #
    # - (ii)  swap two tokens in these positions.                              #
    ############################################################################
    # Replace "..." with your code
    # Replace "..." with your code
    idx1, idx2 = random.sample(range(len(words)), 2)
    words[idx1], words[idx2] = words[idx2], words[idx1]
    new_sentence = ' '.join(words)
    
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################

    return new_sentence

# ========================== Random Insertion ========================== #
def random_insertion(sentence, n):
    
    words = sentence.split()
    new_words = words.copy()
    
    for _ in range(n):
        add_word(new_words)
        
    new_sentence = ' '.join(new_words)
    return new_sentence

def add_word(new_words):
    
    synonyms = []
    ############################################################################
    # TODO: Randomly choose one synonym and insert it into the word list.      #
    # - (i)  Get a synonym word of one random word from the word list;         #
    # - (ii) Insert the selected synonym into a random place in the word list. #
    ############################################################################
    # Replace "..." with your code
    # Replace "..." with your code
    # Replace "..." with your code
    idx = random.randint(0, len(new_words)-1)
    word = new_words[idx]
    syns = get_synonyms(word)
    if syns:
        synonym = random.choice(syns)
        new_words.insert(random.randint(0, len(new_words)), synonym)

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
