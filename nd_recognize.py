"""
ND-RECOGNIZE: Non-Deterministic Finite State Automaton Recognizer
==================================================================

Implements the agenda-based ND-RECOGNIZE algorithm exactly as structured
in Jurafsky & Martin, "Speech and Language Processing", Chapter 2,
Figure 2.21 (page 39): three functions -- ND-RECOGNIZE, GENERATE-NEW-STATES
and ACCEPT-STATE? -- mirrored below with matching names/roles.

The agenda can be configured to behave as either a Stack (DFS) or a
Queue (BFS): this is the single toggle inside NEXT(agenda) that changes
the search order without changing the algorithm's logic at all.

Uses only the Python standard library (collections.deque).
"""

from collections import deque

EPSILON = ""  # internal symbol used to mark epsilon (empty) transitions


# ---------------------------------------------------------------------------
# 1. Automaton Representation
# ---------------------------------------------------------------------------
class NFSA:
 
    def __init__(self, start_state, accept_states):
        self.transition_table = {}
        self.start_state = start_state
        self.accept_states = set(accept_states)

    def add_transition(self, from_state, symbol, to_state):
        self.transition_table.setdefault(from_state, {}).setdefault(symbol, [])
        self.transition_table[from_state][symbol].append(to_state)

    def lookup(self, state, symbol):
        return self.transition_table.get(state, {}).get(symbol, [])

    def is_accept_state(self, state):
        return state in self.accept_states


# ---------------------------------------------------------------------------
# 3. Regular Expression Wrapper
# ---------------------------------------------------------------------------
def compile_regex(pattern):
    """
    Compile a small subset of regex syntax (literal characters and a
    trailing '*' meaning "zero or more of the preceding character") into
    an NFSA, following the style of Figure 2.19 in Jurafsky & Martin
    (the /baa*!/ example).

    Supported patterns: sequences of literal characters, optionally with
    a single '*' applied to the character immediately before it.
    e.g. "baaa!"  -> exact literal chain
         "baa*!"  -> b, a, (a)*, !   (matches baa!, baaa!, baaaa!, ...)
    """
    nfsa = NFSA(start_state="q0", accept_states=[])
    state_counter = 0
    current_state = "q0"

    i = 0
    while i < len(pattern):
        ch = pattern[i]
        starred = (i + 1 < len(pattern) and pattern[i + 1] == "*")

        if starred:
            state_counter += 1
            loop_state = f"q{state_counter}"
            nfsa.add_transition(current_state, ch, loop_state)
            nfsa.add_transition(loop_state, ch, loop_state)      # repeat (>=1 more)

            state_counter += 1
            skip_state = f"q{state_counter}"
            nfsa.add_transition(current_state, EPSILON, skip_state)  # zero occurrences
            nfsa.add_transition(loop_state, EPSILON, skip_state)     # done repeating

            current_state = skip_state
            i += 2  # consumed ch and '*'
        else:
            state_counter += 1
            next_state = f"q{state_counter}"
            nfsa.add_transition(current_state, ch, next_state)
            current_state = next_state
            i += 1

    nfsa.accept_states = {current_state}
    return nfsa


# ---------------------------------------------------------------------------
# 2. Search Strategy Configuration
# ---------------------------------------------------------------------------
# Each agenda item is a search-state tuple: (current_node, tape_pointer)
# -- exactly the "combination of node and tape-position" defined under
# Figure 2.21.

def accept_state(search_state, tape, machine):
    """
    ACCEPT-STATE?(search-state) returns true or false.

    current-node <- the node search-state is in
    index        <- the point on the tape search-state is looking at
    if index is at the end of the tape and current-node is an accept
    state of machine then return true else return false
    """
    current_node, index = search_state
    if index == len(tape) and machine.is_accept_state(current_node):
        return True
    return False


def generate_new_states(current_search_state, tape, machine):
    """
    GENERATE-NEW-STATES(current-state) returns a set of search-states.

    current-node <- the node the current search-state is in
    index        <- the point on the tape the current search-state is
                     looking at
    return a list of search states from transition table as follows:
        (transition-table[current-node, epsilon], index)
        U
        (transition-table[current-node, tape[index]], index + 1)
    """
    current_node, index = current_search_state
    new_states = []

    # (transition-table[current-node, epsilon], index)
    for eps_state in machine.lookup(current_node, EPSILON):
        new_states.append((eps_state, index))

    # (transition-table[current-node, tape[index]], index + 1)
    if index < len(tape):
        symbol = tape[index]
        for next_node in machine.lookup(current_node, symbol):
            new_states.append((next_node, index + 1))

    return new_states


def nd_recognize(tape, machine, strategy="DFS", verbose=True):
    """
    function ND-RECOGNIZE(tape, machine) returns accept or reject

    agenda <- {(Initial state of machine, beginning of tape)}
    current-search-state <- NEXT(agenda)
    loop
        if ACCEPT-STATE?(current-search-state) returns true then
            return accept
        else
            agenda <- agenda U GENERATE-NEW-STATES(current-search-state)
            if agenda is empty then
                return reject
            else
                current-search-state <- NEXT(agenda)
    end

    NEXT(agenda) is the only place the search strategy is decided:
        strategy="DFS" -> agenda behaves as a Stack   (LIFO, deque.pop())
        strategy="BFS" -> agenda behaves as a Queue   (FIFO, deque.popleft())
    """
    agenda = deque()
    agenda.append((machine.start_state, 0))  # {(Initial state, beginning of tape)}

    def next_from_agenda():
        if strategy == "DFS":
            return agenda.pop()        # LIFO -> Stack -> Depth-First
        elif strategy == "BFS":
            return agenda.popleft()    # FIFO -> Queue -> Breadth-First
        else:
            raise ValueError("strategy must be 'DFS' or 'BFS'")

    current_search_state = next_from_agenda()
    steps = 1

    if verbose:
        print(f"\n--- ND-RECOGNIZE running in {strategy} mode on tape '{tape}' ---")
        print(f"Step {steps:2d}: current-search-state = {current_search_state!r} "
              f"| agenda: {list(agenda)}")

    while True:
        if accept_state(current_search_state, tape, machine):
            if verbose:
                print(f"ACCEPT-STATE? -> True. Returning ACCEPT.\n")
            return True, steps

        new_states = generate_new_states(current_search_state, tape, machine)
        agenda.extend(new_states)  # agenda <- agenda U GENERATE-NEW-STATES(...)

        if verbose:
            print(f"          GENERATE-NEW-STATES -> {new_states} "
                  f"| agenda now: {list(agenda)}")

        if not agenda:
            if verbose:
                print(f"Agenda is empty. Returning REJECT.\n")
            return False, steps

        current_search_state = next_from_agenda()
        steps += 1

        if verbose:
            print(f"Step {steps:2d}: current-search-state = {current_search_state!r} "
                  f"| agenda: {list(agenda)}")


# ---------------------------------------------------------------------------
# 4. Execution Log and Comparison
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pattern = "ab*c*!"
    target_string = "abbccc!"

    machine = compile_regex(pattern)

    print("=" * 70)
    print(f"NFSA compiled for pattern: {pattern!r}")
    print(f"Target string to recognize: {target_string!r}")
    print("=" * 70)

    dfs_accepted, dfs_steps = nd_recognize(target_string, machine, strategy="DFS")
    bfs_accepted, bfs_steps = nd_recognize(target_string, machine, strategy="BFS")

    print("=" * 70)
    print("COMPARISON: DFS vs BFS")
    print("=" * 70)
    print(f"DFS -> accepted={dfs_accepted}, steps taken={dfs_steps}")
    print(f"BFS -> accepted={bfs_accepted}, steps taken={bfs_steps}")
    if dfs_steps != bfs_steps:
        faster = "DFS" if dfs_steps < bfs_steps else "BFS"
        print(f"{faster} reached acceptance in fewer search steps on this input.")
    else:
        print("Both strategies took the same number of steps on this input.")