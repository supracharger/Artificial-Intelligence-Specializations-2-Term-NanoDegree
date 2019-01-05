import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Implemented the recognizer
    probabilities = []
    all_word_Xlengths = test_set.get_all_Xlengths()     # Get SignWords & Lens
    # Loop by SignWords
    for wordLabel in all_word_Xlengths:
        signWord, lens = all_word_Xlengths[wordLabel]               # Get Cordinates & array len
        D = {}                                                      # Dictionary for each word
        probabilities.append(D)                                     # Append Dict. to List
        # Loop each HMM acording to each word in DataFrame to get LogL for that Word
        for label in models:
            try: D[label] = models[label].score(signWord, lens)     # Get LogL for that Word HMM
            except: 
                D[label] = float('-inf')                            # Invalid Score: Save value that will never be in a guess
                continue                                            # Invalid Score: CONTINUE LOOP
    # Find the Max LogLikelyhood in Dict for each word, & (BestGuess:str)Save the index of the Highest LogL.
    GetValue = lambda v: v[1]               # For Max(): Use the Value of the Dict, Not the Key
    # Best Guess for Words: Highest LogL
    guesses = [max(D.items(), key=GetValue)[0] for D in probabilities]
    # Return Probabilities, Guesses
    return probabilities, guesses
