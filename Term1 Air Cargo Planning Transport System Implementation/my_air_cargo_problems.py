from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        # TODO create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """

            # TODO create all load ground actions from the domain Load action
            loads = []
            for c in self.cargos:               # Loop all Cargos
                for p in self.planes:           # Loop all Planes
                    for a in self.airports:     # Loop all Airports
                        # Cited: Structure of this block re-coded from fly_actions()
                        # True/ Not Preconditions
                        precond_pos = [expr("At({}, {})".format(c, a)),            
                                       expr("At({}, {})".format(p, a))]
                        precond_neg = []                                            
                        # True/ Not Effects
                        effect_add = [expr("In({}, {})".format(c, p))]             
                        effect_rem = [expr("At({}, {})".format(c, a))]            
                        load = Action(expr("Load({}, {}, {})".format(c, p, a)),         # Create Load Action
                                        [precond_pos, precond_neg],
                                        [effect_add, effect_rem])
                        loads.append(load)                                              # Add to list
            return loads

        def unload_actions():
            """Create all concrete Unload actions and return a list

            :return: list of Action objects
            """
            
            # TODO create all Unload ground actions from the domain Unload action
            unloads = []
            for c in self.cargos:               # Loop all Cargos
                for p in self.planes:           # Loop all Planes
                    for a in self.airports:     # Loop all Airports
                        # Cited: Structure of this block re-coded from fly_actions()
                        # True/ Not Preconditions
                        precond_pos = [expr("In({}, {})".format(c, p)),
                                       expr("At({}, {})".format(p, a))]
                        precond_neg = []
                        # True/ Not Effects
                        effect_add = [expr("At({}, {})".format(c, a))]
                        effect_rem = [expr("In({}, {})".format(c, p))]
                        unload = Action(expr("Unload({}, {}, {})".format(c, p, a)),     # Create Unload Action
                                        [precond_pos, precond_neg],
                                        [effect_add, effect_rem])
                        unloads.append(unload)                                          # Add to list
            return unloads

        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """

        # Decode Bool string into its conditions
        fState = decode_state(state, self.state_map)

        # Get All Possible Actions
        possible_actions = []
        # Loop all Actions
        for action in self.actions_list:
            iter = 0
            # Loop & Check All Pos./ Neg. Precond. of that Action
            # Loop 1st: Check Pos. precond; Loop 2nd: Check Neg. precond;
            while iter<2 and iter>=0:
                if iter==0:         # Pos. Precond.
                    precond = action.precond_pos
                    stateCond = fState.pos
                else:               # Neg. Precond.
                    precond = action.precond_neg
                    stateCond = fState.neg
                iter += 1       # INC to go to next Set of Precond.
                # Loop all Precond. in set
                for cond in precond:
                    # If Precond. Not found: break & Continue to next "action"
                    if not cond in stateCond:
                        iter = -1       # -1: values that shows it did Not pass all Conds.
                        break
            # If Passed all Cond: add append Action. >0: Passed all Conds.
            if iter>0:
                possible_actions.append(action)

        # RETURN
        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """

        # Decode Bool string into its conditions
        fState = decode_state(state, self.state_map)
        # Create NewState with the same Conds. as the current
        #   - !copy(): decode_state() already creates new lists
        new_state = FluentState(fState.pos, fState.neg)

        # Loop:: iter=0: True Effects; iter=1: Not Effects
        for iter in range(2):
            # Using True Effects
            if iter==0:
                actEft = action.effect_add          # True Effects
                stateEftAdd = new_state.pos         # Add New State Pos.
                stateEftRemove = new_state.neg      # Remove New State Neg.
            # Using Not Effects
            else:
                actEft = action.effect_rem          # Not Effects
                stateEftAdd = new_state.neg         # Add New State Neg.
                stateEftRemove = new_state.pos      # Remove New State Pos.
            # Loop by Effects of that Set
            for effect in actEft:
                stateEftAdd.append(effect)          # NewState Add Effect to same Polar sign Set
                stateEftRemove.remove(effect)       # NewState Remove Effect from opposite Polar sign Set
        # Encode into Bool String
        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        self.h_ignore_preconditions(node)
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.

        RETURNS: "the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions"
        """
        
        # Get the Pos. Conditions of the Current State
        pos = decode_state(node.state, self.state_map).pos

        # Loop by Goal Conds. to figure out how many are Not Satisfied.
        #   - "Goal Conds. Not Satisfied" = "Minimum number of Actions"
        # Implemented (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)
        count = 0
        for cond in self.goal:
            if not cond in pos: count += 1      # INC if current Goal is Not Statisfied
        # RETURNS: the minimum number of actions to reach Goal
        return count

def NegFluents(pos: list, cargos: list, planes: list, airports: list) -> list:
    neg = []
    # Loop by True Precond.
    for cond in pos:
        if cond.op!='At': raise Exception()         # Make sure it is only the At()
        args = [a.op for a in cond.args]            # Get the 2 args
        # Find the objects list that has the secondArg in that list
        items = []
        if args[1] in airports: items = airports 
        elif args[1] in planes: items = planes
        elif args[1] in cargos: items = cargos
        else: raise Exception()
        # Add other items in the NotClause to corispond with the 1stArg
        # Loop by labels in items
        for i in items:
            if i==args[1]: continue                                 # Do Not add the same item
            neg.append(expr("%s(%s, %s)" % (cond.op, args[0], i)))  # Append

    # Add Cargo Not in any Planes: In() Clause
    neg.extend([expr("In(%s, %s)" % (c, p)) for c in cargos for p in planes])
    # RETURN
    return neg


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    NegFluents(pos, cargos, planes, airports)
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    # TODO implement Problem 2 definition
    # Items
    cargos = ['C1', 'C2', 'C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['JFK', 'SFO', 'ATL']
    # Preconditions
    pos = [expr('At(C1, SFO)'), 
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           expr('At(P3, ATL)'),
           ]
    # Create Neg. Fluents
    neg = NegFluents(pos, cargos, planes, airports)
    # Create FluentState
    init = FluentState(pos, neg)
    # Goal
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    # Problem 3 definition
    # Items
    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO', 'ATL', 'ORD']
    # Preconditions
    pos = [expr('At(C1, SFO)'), 
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(C4, ORD)'), 
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
            ]
    # Create Neg. Fluents
    neg = NegFluents(pos, cargos, planes, airports)
    # Create FluentState
    init = FluentState(pos, neg)
    # Goal
    goal = [expr('At(C1, JFK)'),
            expr('At(C3, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C4, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)

if __name__ == "__main__":
    air_cargo_p1()
    print()
