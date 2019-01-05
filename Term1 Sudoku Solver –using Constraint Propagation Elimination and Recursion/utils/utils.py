rows = 'ABCDEFGHI'
cols = '123456789'

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [s+t for s in A for t in B]

def create_units_peers(unitlist):
    """
    Creates Lists Units & Peers
    Args: None
    Returns:
        units, peers
    """
    units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
    peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)
    return units, peers

# Create Sudoku Box Labels
boxes = cross(rows, cols)

# UNITS ...........................................
row_units = [cross(r, cols) for r in rows]
# row_units[0] = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']
# This is the top most row.
col_units = [cross(rows, c) for c in cols]
# column_units[0] = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'I1']
# This is the left most column.
square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]
# square_units[0] = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
# This is the top left square.

# Create a List of Units: unitlist
unitlist = row_units + col_units + square_units
# Create Lists units & peers
units, peers = create_units_peers(unitlist)