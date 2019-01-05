# utils.py ___________________________________________________________________________________________
rows = 'ABCDEFGHI'
cols = '123456789'

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [s+t for s in A for t in B]

# Create Sudoku Box Labels
boxes = cross(rows, cols)

# UNITS ...........................................
row_units = [cross(r, cols) for r in rows]
col_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]
# Diagonal Units
diagonal_units = [[a+str(i+1) for i,a in enumerate(rows)]]                  # Diag. 1
diagonal_units.append([a+str(len(rows) - i) for i,a in enumerate(rows)])    # Diag. 2

# Create a List of Units: unitlist
unitlist = row_units + col_units + square_units + diagonal_units
# Create Lists units & peers
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)
# End of utils.py __________________________________________________________________________________

# Validation _______________________________________________________________________________________
# if Solution Passes nothing will be printed to console
def validateSolution(values, addUnitGroups = False, addNames = False):
    unitGroups = [row_units, col_units, square_units]
    names = ['row_units', 'col_units', 'square_units']
    erCtr = 0
    lines = []

    # Exit if No Values
    if not values: return values

    # Append other Unit Groups to List & there corresponding Names. Like Diagonal Units
    if (addUnitGroups):
        assert addNames and len(addUnitGroups)==len(addNames), "ERROR!: there should be a name corresponding to each group."
        unitGroups.extend(addUnitGroups)
        names.extend(addNames)

    # Check there is 1 Value per box
    unfinished = [box for box, v in values.items() if len(v)>1]
    lines.append("\n Validated Solution:")
    if len(unfinished)>0:
        for l in lines: print(l)
        print("\tThere are unfinished boxes in solution(%d): %s \n" % (len(unfinished), str(unfinished)))
        return False

    # Loop by unitType i.e. row_units, col_units, ...
    for g, unitType in enumerate(unitGroups):
        # Each Unit
        for u, unit in enumerate(unitType):
            # Get the values of each box in the unit
            boxVals = [values[box] for box in unit]
            # Loop by each number 1-9
            for num in cols:
                # If it Cannot find number in sequence prompt user
                if not num in boxVals:
                    lines.append("\tFailed in %s %d Cannot find num: %s" % (names[g], u, num))
                    erCtr += 1
                    break
    
    # If there ARE Errors
    if erCtr>0:
        for l in lines: print(l)
        print("\t=> Errors: %d \n" % erCtr)
        return False

    return values

class Test:
    def __init__(self):
        self._savedVals = []
        self._sCtr = -1

    def saveValues(self, values):
        self._sCtr += 1
        if (values): self._savedVals.append(values.copy())
# End of Validation ________________________________________________________________________________

assignments = []

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """

    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """

    twins = []

    # Find all instances of naked twins
    # Loop by Each unit
    for i, unit in enumerate(unitlist):
        # Find Twins in unit: with a Length of 2, and the same possible values
        twinsAdd = [tuple(sorted((a,b))) for a in unit for b in unit 
                    if len(values[a])==2 and values[a]==values[b] and a != b]
        twinsAdd = set(tuple(twinsAdd))         # Get unique set of twins for that unit
        twins.append(twinsAdd)                  # Append to Twin List

    # Get unmutated values for twins, make sure the value is the same & not changed during updating the boxes in the below loop
    uniqTwins = set(tuple(box for twinsUnit in twins for t in twinsUnit for box in t))  # Unroll to a list of boxes
    TwinsVal = {box:values[box] for box in uniqTwins}

    # Eliminate the naked twins as possibilities for their peers
    # Loop by twins in unit
    for i, unitTwins in enumerate(twins):
        # Loop all unit twins
        for t in unitTwins:
            v = TwinsVal[t[0]]                          # Get value of the twins. Only 1st twin is needed to get the value
            # Loop each Box in a unit
            for box in unitlist[i]:
                if values[box]==v: continue                     # For Twins do Not Replace Possible values
                val = values[box]                               # Temp. value to modify & re-assign
                for d in v: val = val.replace(d, '')            # Remove the 2 twins values if found
                assign_value(values, box, val)  
    return values

def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    
    assert len(grid) == 81, "grid must have a input len of 81."
    
    dPuzzle = {box:'' for box in boxes}    # Dict. form of Puzzle
    pix = -1        # Increment Index
    # Loop Rows
    for r in row_units:
        # Loop Cols in Row
        for k in r:
            pix += 1
            assert grid[pix]=='.' or grid[pix] in cols, "Invalid key in grid: " + grid[pix]
            # if not empty value: save constant number; else if empty box: assign all poss. numbers 1-9
            val = grid[pix] if (grid[pix] != '.') else '123456789'  # Get Value
            assign_value(dPuzzle, k, val)                           # Assign value to box
    return dPuzzle

def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    
    # NOTE!: Copied directly from Udacity since the function was given
    #       and not asked to create on my own -Andrew

    # Validation -Andrew
    if not values:
        print ("display() Failed: Sudoku Grid could Not be Solved, values: %s" % str(values))
        return

    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return

def eliminate(values):
    """Eliminate values using the Eliminate strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with Eliminate applied.
    """
    # Loop each Box
    for kbox in values:
        v = values[kbox]            # Value to Remove
        if (len(v) > 1): continue   # Loop only Constant Values, Not Variable
        peerOne = peers[kbox]       # List of Peers for current box
        for p in peerOne: 
            assign_value(values, p, values[p].replace(v, ''))   # Remove from possible values
    return values

def only_choice(values):
    """Eliminate values using the Only Choice strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the Only Choice eliminated from peers.
    """

    # Loop by Unit Group
    for unit in unitlist:
        # Tally Only_choice num; '': Not found; '*': more than one contain that val; 
        #   (otherVal): Only Choice box key/Id
        tally = {n:'' for n in '123456789'}
        # Loop Each box
        for box in unit:
            for d in values[box]:
                # Assign it boxID if none assigned, if contains box ID assign '*' for multiple
                tally[d] = box if tally[d]=='' else '*'
        # If Only 1 choice make it that value
        for num, box in tally.items():
            if box != '' and box != '*':
                assign_value(values, box, num)      # Assign it only choice num.
    return values

def reduce_puzzle(values):
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])

        # Use the Eliminate Strategy
        values = eliminate(values)
        # Use the Only Choice Strategy
        values = only_choice(values)
        # Use the Naked Twins Strategy
        values = naked_twins(values)

        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

def search(values):
    "Using depth-first search and propagation, create a search tree and solve the sudoku."

    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)
    if (not values): return False                                               # Failed Puzzle
    elif len([b for b in boxes if len(values[b])==1]) == 81: return values      # Solved!!
    
    # Choose one of the unfilled squares with the fewest possibilities
    minList = [(len(values[b]), b) for b in boxes if len(values[b]) > 1]
    minList.sort()          # Sort by fewest poss, maybe usefull in the future 
    minB = minList[0][1]    # Select 1st

    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    for v in values[minB]:
        newGrid = values.copy()
        assign_value(newGrid, minB, v)  # Assign it the value
        trial = search(newGrid)         # used this from solution.py
        if trial: return trial          

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """

    # Convert String Grid to Dictionary Sudoku Values
    values = grid_values(grid)
    # Solve Sudoku
    values = search(values)
    # Validate Solution: Returns False if there are broken rules
    values = validateSolution(values, [diagonal_units], ['diagonal_units'])

    return values

if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
