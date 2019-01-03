import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        Cited Help: https://discussions.udacity.com/t/help-getting-started-with-selector-code-in-model-selectors/397770/2?u=suprachargers2d2n

        :return: GaussianHMM object
        """

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Implement model selection based on BIC scores
        bestBic = float('inf')
        best = None
        features = len(self.sequences[0][0])    # Num. of Features
        # Loop Components from Min to Max
        for stateLen in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(stateLen)                               # Get & Train Model
            if not model: continue                                          # Invalid Model Continue Loop
            # Calc BIC
            params = stateLen**2 + 2 * stateLen * features - 1
            try: logL = model.score(self.X, self.lengths)                        # Current Log Likelyhood
            except: continue
            bic = -2 * logL + params * math.log(len(self.sequences))
            # Save Best Model: lowest BIC
            if bic<bestBic:
                bestBic = bic
                best = model
        # Return Best BIC
        return best


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        # Cited Help: https://discussions.udacity.com/t/dic-score-calculation/238907/2
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Get All Other Words Seq. Data & Lengths in otherWords & otherLens accordingly
        otherWords = []
        otherLens = []
        for label in self.hwords.keys():
            otherWords.append(self.hwords[label][0])
            otherLens.append(self.hwords[label][1])
        def GetDIC(model):
            otherList = []
            # Try Current Log Likelyhood
            try: logL = model.score(self.X, self.lengths)        
            except: return None
            # Try Each Other Word Log Likelyhood & append Score to list
            for i, other in enumerate(otherWords):
                try: otherList.append(model.score(other, otherLens[i]))
                except: pass 
            # Final DIC Calc
            return logL - np.mean(otherList)
                
        # Implement model selection based on DIC scores
        bestDic = float('-inf') 
        best = None
        # Loop Components from Min to Max
        for stateLen in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(stateLen)       # Get & Train Model
            if not model: continue                  # Invalid Model Continue Loop 
            # Calc DIC
            dic = GetDIC(model)
            if dic==None: continue                  # Exit: Could Not get LogL of Current Word
            # Save Best Model: highest dic
            if dic>bestDic:
                bestDic = dic
                best = model
        # Return Best Model
        return best


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Implemented model selection using CV
        i = -1
        best = None                                                         # Best Model
        bestCV = float('inf')                                               # Best Model Score
        stateLens = range(self.min_n_components, self.max_n_components + 1) # Num. of States to Test HMM
        Fold = KFold()                                                      # KFold Obj.
        Fold.n_splits = 2 if len(self.sequences)>1 else 1                   # If enouph samples: Num Splits in the DataSet (2 Train & Test).
        # Loop over diff. State Num for HMM with Cross Val (Train & Test)
        for trainIdx, testIdx in Fold.split(self.sequences):
            i += 1                                                          # Index
            train = combine_sequences(trainIdx, self.sequences)             # Train Set
            test = combine_sequences(testIdx, self.sequences)               # Test Set
            model = self.baseSetModel(stateLens[i], train[0], train[1])     # Model Trained on Training Set
            if not model: continue
            try: cv = model.score(test[0], test[1])                         # Model Scored on Test Set
            except: continue
            # Get Best CV Model: Lowest CV
            if cv<bestCV:
                bestCV = cv
                best = model
        # Return Best Model Cross Validation
        return best

    def baseSetModel(self, num_states, X, Lens):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, Lens)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None
