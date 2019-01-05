# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: 
We use Constraint Propagation to solve the Naked Twins problem by using the process below. First, we find all known twins in a given Unit (Row, Column, Square, and Diagonal Units). A Naked Twin is defined by each twin set having equal values, and a length of two. Then, for all known twins we remove the value (the two possibilities) of the twins from the other boxes possibilities in a given Unit. In conclusion, since there are two twins with equal values they must each have one of those values, and it is not possible for other boxes to have those twin values; all which is a Constraint Propagation heuristic, and aids in reducing the search space.

Constraint Propagation in General to Naked Twins:
We use Constraint Propagation to solve the Naked Twins problem by using the constraint of the Naked Twins rule itself. The Naked Twins heuristic minimizes the search space by a certain amount by using the known rule to minimize or constrain the problem (Constraint Propagation). Combined with the others “Elimination” and “One Choice,” and the use of Recursion, the search space is greatly limited. Finally, Constraint Propagation and the use of the heuristic Naked Twins minimizes the problem and reduces CPU runtime for other “Brute Force” Primal methods (Depth First Search) that would usually take a significant time to complete.

# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A: 
We use Constraint Propagation to solve the Diagonal Sudoku problem by adding the Diagonal constraint to the Sudoku Units rule. The Diagonal rule adds considerable complexity, and increases the search space since a new constraint is added. To solve this dilemma the solution is actually quite simple. If one thinks back to a Regular Sudoku, the only difference from it and the Diagonal one is the added Unit group “Diagonal Units” which contains the two major diagonals of the board. In conclusion, Modifying “solutions.py” by adding the “Diagonal Units” to “unitlist” variable, and making sure that “units” and “peers” variables are updated properly with the “Diagonal Units” (No modification would be needed to solutions.py) the problem will be solved. Additionally, by adding the two “Diagonal Units” to “unitlist”, and making sure “units” and “peers” variables are updated accordingly with the diagonal units, one can modify and solve the regular Sudoku Problem with the added constraint of the Diagonal Sudoku rule (The Diagonal Sudoku).

Constraint Propagation in General to Diagonal Units:
We use Constraint Propagation to solve the Diagonal Sudoku problem to minimize the search space. Since using the Diagonal Sudoku over the regular Sudoku significantly increases the complexity and search space. One must ideally use other methods such as Constraint Propagation to minimize the solution. Constraint Propagation limits the space by using known heuristics, along with Recursion, to not completely solve the problem, but to limit it significantly. Finally, it can minimize the space significantly by minimizing CPU runtime until moving to more Primal methods like Depth First Search.

### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solution.py` - Fill in the required functions in this file to complete the project.
* `test_solution.py` - You can test your solution by running `python -m unittest`.
* `PySudoku.py` - This is code for visualizing your solution.
* `visualize.py` - This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the `assign_value` function provided in solution.py

### Submission
Before submitting your solution to a reviewer, you are required to submit your project to Udacity's Project Assistant, which will provide some initial feedback.  

The setup is simple.  If you have not installed the client tool already, then you may do so with the command `pip install udacity-pa`.  

To submit your code to the project assistant, run `udacity submit` from within the top-level directory of this project.  You will be prompted for a username and password.  If you login using google or facebook, visit [this link](https://project-assistant.udacity.com/auth_tokens/jwt_login) for alternate login instructions.

This process will create a zipfile in your top-level directory named sudoku-<id>.zip.  This is the file that you should submit to the Udacity reviews system.


