"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

# Finds the distance from center for each value in 'allMoves' as a percentage of maxDistanceFromCenter
# & Averages the percentages.
def distanceFromCenter(game, allMoves):
    # Cited Idea Distance from center Function: https://discussions.udacity.com/t/how-is-the-distance-from-the-center-of-the-board-a-good-heuristic-function/436160
    # However, the code was created by myself.

    # Exit & Return zero if No Moves
    if not allMoves: return 0

    # Get Center
    center = ((game.width-1)/2, (game.height-1)/2)
    # Loop Moves
    sum = 0
    ctr = 0
    for move in allMoves:
        # Sum: Take the Avg. Distance of x & y. Distance is in a %percent of maxDistance
        sum += (abs((move[0]-center[0])/center[0]) + abs((move[1]-center[1])/center[1])) / 2        
        ctr += 1
    # Return Average
    return sum / ctr

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    legalMoves = game.get_legal_moves()         # Get Legal Moves
    maxMovesLen = game.height * game.width      # Get Number of max moves for 'lenMoves' to have a result that is a percentage.

    # For each 'legalMoves' get the distance from center percent & Average them. On a scale 0 => 1.
    centerEval = distanceFromCenter(game, legalMoves)
    # Get Len of number of Moves as a percentage. On a scale 0 => 1.
    # *10: Multiplier that increases & normalizes the value to be equal with 'centerEval.' Multiplier is Calc'd by avg. of all 
    #       the values of 'lenMoves' & 'centerEval' over the whole tournament.
    lenMoves = 10 * (len(legalMoves) / maxMovesLen)

    # *2: Weight 'lenMoves' twice over 'centerEval'
    return float(2 * lenMoves + centerEval)

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Simple Evaluation Function is based on the number of moves available.
    return float(len(game.get_legal_moves()))

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # My own Evaluation Function, thinking it would be better maximize my own moves rather than the opponent.
    return float(2 * len(game.get_legal_moves()) - len(game.get_legal_moves(game.inactive_player)))

# Do Not Modify this class
class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def min_value(self, game, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """

        # Timer Check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Get Legal Moves
        legalMoves = game.get_legal_moves()

        # If Reached Max depth or at Leaf Node, and get the Score
        # Cited implementaintion "depth==0": https://discussions.udacity.com/t/cant-figure-out-why-agents-are-forfeiting/443454?u=suprachargers2d2n
        if len(legalMoves)==0 or depth==0:
            return self.score(game, self)   # Return score of Leaf/Deepest node

        v = float('inf')           # Value that will always be overwritten
        # Loop each Move in next branch from legalMoves
        for move in legalMoves:
            score = self.max_value(game.forecast_move(move), depth-1)   # Get the Score of the Branch
            v = min(v, score)                                           # Get the Min Score

        return v 

    def max_value(self, game, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """

        # Timer Check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Get Legal Moves
        legalMoves = game.get_legal_moves()

        # If Reached Max depth or at Leaf Node, and get the Score
        # Cited implementaintion "depth==0": https://discussions.udacity.com/t/cant-figure-out-why-agents-are-forfeiting/443454?u=suprachargers2d2n
        if len(legalMoves)==0 or depth==0:
            return self.score(game, self)

        v = float('-inf')          # Value that will always be overwritten
        # Loop each Move in next branch from legalMoves
        for move in legalMoves:
            score = self.min_value(game.forecast_move(move), depth-1)   # Get the Score of the Branch
            v = max(v, score)                                           # Get the Max Score

        return v

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                    functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Get Legal Moves
        legalMoves = game.get_legal_moves()

        # If less than 1 move
        if len(legalMoves)<=1:
            if len(legalMoves)==1: return legalMoves[0]
            return (-1, -1)

        bestMove = legalMoves[0]
        v = float('-inf')      # Init. Move that will always be overwritten
        # Loop each Move in next branch from legalMoves
        for move in legalMoves:
            score = self.min_value(game.forecast_move(move), depth-1)   # Get the Score of the Branch
            if score > v:                                               # Get the Max Score
                v = score                                                   # Max Score
                bestMove = move                                             # Best Move According to Max Score

        # Check for Infinity (Check for end game): if true return 'No Move'
        if v==float('inf') or v==float('-inf'): return (-1, -1)

        return bestMove


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def min_value(self, game, depth, alpha, beta):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """

        # Timer Check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Get Legal Moves
        legalMoves = game.get_legal_moves()

        # If Reached Max depth or at Leaf Node, and get the Score
        # Cited implementaintion "depth==0": https://discussions.udacity.com/t/cant-figure-out-why-agents-are-forfeiting/443454?u=suprachargers2d2n
        if len(legalMoves)==0 or depth==0:
            return self.score(game, self)

        v = float('inf')           # Value that will always be overwritten
        # Loop each Move in next branch from legalMoves
        for move in legalMoves:
            score = self.max_value(game.forecast_move(move), depth-1, alpha, beta)  # Get the Score of the Branch
            v = min(v, score)                                                       # Get the Min Score
            if v<=alpha: return v                                                   # Crop Nodes according to alpha
            beta = min(beta, v)                                                     # Update min Beta Threshold when applicable

        return v 

    def max_value(self, game, depth, alpha, beta):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """

        # Timer Check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Get Legal Moves
        legalMoves = game.get_legal_moves()

        # If Reached Max depth or at Leaf Node, and get the Score
        # Cited implementaintion "depth==0": https://discussions.udacity.com/t/cant-figure-out-why-agents-are-forfeiting/443454?u=suprachargers2d2n
        if len(legalMoves)==0 or depth==0:
            return self.score(game, self)

        v = float('-inf')          # Value that will always be overwritten
        # Loop each Move in next branch from legalMoves
        for move in legalMoves:
            score = self.min_value(game.forecast_move(move), depth-1, alpha, beta)  # Get the Score of the Branch
            v = max(v, score)                                                       # Get the Max Score
            if v >= beta: return v                                                  # Crop Nodes according to Beta
            alpha = max(alpha, v)                                                   # Update max Alpha Threshold when applicable

        return v

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Default best Move: which is No move found
        best_move = (-1, -1)

        # The try/except block will automatically catch the exception
        # raised when the timer is about to expire.
        try:
            # Iterative Deepening
            depth = self.search_depth - 1    # -1: So I can increment at the beginging of the loop
            #depth = 0
            while(True):
                depth += 1                                      # increase depth for iterative Deepening
                #if depth==2: continue                           # So on 2nd iteration it skips over a depth of 2, & goes to 3 to increase intelligence
                best_move = self.alphabeta(game, depth)         # Retrieve best move for current depth
                if best_move==(-1, -1): break                   # if no move found: end search. Helps with 100's of games run time.

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        # Timer Check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Get Legal Moves
        legalMoves = game.get_legal_moves()

        # If less than 1 move
        if len(legalMoves)<=1:
            if len(legalMoves)==1: return legalMoves[0]
            return (-1, -1)

        bestMove = legalMoves[0]
        v = float('-inf')      # Init. Move that will always be overwritten
        # Loop each Move in next branch from legalMoves
        for move in legalMoves:
            score = self.min_value(game.forecast_move(move), depth-1, alpha, beta)  # Get the Score of the Branch
            if score > v:                                                           # Get the Max Score
                v = score                                                           # Max Score
                bestMove = move                                                     # Best Move According to Max Score
            if v >= beta: return v                                                  # Crop Nodes according to Beta
            alpha = max(alpha, v)                                                   # Update max Alpha Threshold when applicable

        # Check for Infinity (Check for end game): if true return 'No Move'
        if v==float('inf') or v==float('-inf'): return (-1, -1)

        return bestMove

if __name__ == '__main__':
    #A = AlphaBetaPlayer(timeout=500)
    #move = A.get_move()
    #agent = Agent()
    
    #iso = Isolation(OldBoardState())
    #MM = MinimaxPlayer(2, custom_score)
    #move = MM.get_move(iso.Board, lambda: 11)

    for k in items:
        if len(items[k])==0: break
        print(items[k])
    print(move)
