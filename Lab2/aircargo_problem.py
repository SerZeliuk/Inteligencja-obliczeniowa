from collections import deque

import pandas as pd
from aircargo import Strips, STRIPS_domain, Planning_problem
import time

def create_air_cargo_domain(cargos, planes, airports):
    
    feature_domain_dict = {}
    
    for c in cargos:
        feature_domain_dict[f"At_{c}"] = set(airports) | set(planes)
    
    for p in planes:
        feature_domain_dict[f"At_{p}"] = set(airports)
   
    for p in planes:
        feature_domain_dict[f"Empty_{p}"] = {True, False}
    
   
    actions = set()

    for c in cargos:
        for p in planes:
            for a in airports:
                name = f"LOAD_{c}_onto_{p}_at_{a}"
                preconds = {
                    f"At_{c}": a,  # ładunek c jest na lotnisku a
                    f"At_{p}": a,   # samolot p jest również na lotnisku a
                    f"Empty_{p}": True  # samolot p jest pusty
                }
                effects = {
                    f"At_{c}": p,    # w efekcie ładunek c będzie w samolocie p
                    f"Empty_{p}": False  # samolot p staje się niepusty
                }
                actions.add(Strips(name, preconds, effects))

    
    for c in cargos:
        for p in planes:
            for a in airports:
                name = f"UNLOAD_{c}_from_{p}_at_{a}"
                preconds = {
                    f"At_{c}": p,  # ładunek c jest w samolocie p
                    f"At_{p}": a,   # samolot p jest na lotnisku a
                    f"Empty_{p}": False,
                }
                effects = {
                    f"At_{c}": a,   # ładunek c trafia na lotnisko a
                    f"Empty_{p}": True  # samolot p staje się pusty
                }
                actions.add(Strips(name, preconds, effects))

    
    
    for p in planes:
        for fr in airports:
            for to in airports:
                if to != fr:
                    name = f"FLY_{p}_from_{fr}_to_{to}"
                    preconds = {
                        f"At_{p}": fr  # samolot p musi być na lotnisku fr
                    }
                    effects = {
                        f"At_{p}": to  # po akcji samolot jest na lotnisku to
                    }
                    actions.add(Strips(name, preconds, effects))

    return STRIPS_domain(feature_domain_dict, actions)


def goal_satisfied(state, goal):
    return all(state.get(k) == v for k, v in goal.items())


import math

CURRENT_GOAL = {}

def do_heuristics(state, action):
    """
    Return False if we want to prune (skip) this action,
    True otherwise.

    'state': dictionary of feature->value, e.g. { "At_C1": "SFO", "At_P1": "JFK", ... }
    'action': a Strips object with:
        - action.name (e.g., "LOAD_C1_onto_P1_at_SFO")
        - action.preconds (dict of feature->value)
        - action.effects (dict of feature->value)

    We also use the global CURRENT_GOAL dictionary to see where
    each cargo/plane ultimately needs to end up.
    """
    name_parts = action.name.split("_")
    action_type = name_parts[0] 

    if action_type in ("LOAD", "UNLOAD"):
        
        cargo_str = name_parts[1]  
        cargo_goal = CURRENT_GOAL.get(f"At_{cargo_str}")

        if cargo_goal is not None:
            # Check the current location of that cargo in 'state'.
            cargo_loc = state.get(f"At_{cargo_str}")
            # If the cargo is already at its goal (and not in a plane),
            # then we do NOT want to move it away.
            # That means skip LOADing or UNLOADing from anywhere if it's already
            # sitting in the correct airport.
            if cargo_loc == cargo_goal:
                # The cargo is at its goal location, so don't load/unload it.
                return False

    if action_type == "FLY":
        # Example action name: FLY_P2_from_JFK_to_ORD
        # name_parts = ["FLY", "P2", "from", "JFK", "to", "ORD"]
        plane_str = name_parts[1]   
        plane_goal = CURRENT_GOAL.get(f"At_{plane_str}")

        if plane_goal is not None:
            plane_loc = state.get(f"At_{plane_str}")
            if plane_loc == plane_goal:
                # The plane is already at its goal. Only allow flying away if
                # it's *carrying or going to pick up cargo* that needs to move.
                #
                # 1) If the plane is NOT empty or about to load something
                #    that is not in its goal location, we might allow it.
                # 2) If everything is at its goal, there's no point moving.
                #
                # A simpler approach is: if the plane is at its goal,
                # just do NOT fly anywhere. That alone can prune some silly steps.
                #
                # We'll do the simpler approach for clarity:
                return False

    return True

def apply_action(state, action, use_heuristics):
    """
    Try to apply 'action' to 'state'. If use_heuristics is True, use the modified
    heuristic to decide whether to prune this action.
    """
    # First, check preconditions
    for k, v in action.preconds.items():
        if state.get(k) != v:
            return None
            
    
    if use_heuristics:
        if not do_heuristics(state, action):
            return None
    
    # If the action passes, create and return the new state with effects applied.
    new_state = state.copy()
    for k, v in action.effects.items():
        new_state[k] = v
    return new_state


def plan_with_timeout(problem: Planning_problem, use_heuristics: bool, max_time=300):
    global CURRENT_GOAL
    CURRENT_GOAL = problem.goal
    start_time = time.time()
    start_state = problem.initial_state
    frontier = deque([(start_state, [])])
    visited = set()

    while frontier:
        
        elapsed = time.time() - start_time
        if elapsed > max_time:
            return None

        state, path = frontier.popleft()
        state_key = tuple(sorted(state.items()))
        if state_key in visited:
            continue
        visited.add(state_key)

        if goal_satisfied(state, problem.goal):
            return path

        for action in problem.prob_domain.actions:
            new_state = apply_action(state, action, use_heuristics)
            if new_state is not None:
                frontier.append((new_state, path + [action]))

    return None 


def generate_all_problems():
    """
    Returns a list of (title, Planning_problem) tuples.
    We group them as:
      1..3: 'Normal' problems with >= 50 states.
      4..6: Problems with multiple subgoals.
      7..9: Larger problems with multiple subgoals that
            likely require ~30+ actions to solve.
    """
    problems = []

    # ------------------------------------------------
    # 1) Normal Problem A: (2 cargo, 2 planes, 3 airports)
    #    => (3 + 2)^2 = 5^2=25 possible cargo placements * 3^2=9 plane placements => 225 total states
    # ------------------------------------------------
    cargos = ["C1", "C2"]
    planes = ["P1", "P2"]
    airports = ["SFO", "JFK", "ORD"]

    dom_1 = create_air_cargo_domain(cargos, planes, airports)
    init_1 = {
        "At_C1": "SFO",
        "At_C2": "ORD",
        "At_P1": "SFO",
        "At_P2": "JFK",
        "Empty_P1": True,      # ensure plane P1 is empty
        "Empty_P2": True       # ensure plane P2 is empty
    }
    goal_1 = {
        "At_C1": "JFK",
        "At_C2": "SFO"
    }
    prob_1 = Planning_problem(dom_1, init_1, goal_1)
    problems.append(("Normal Problem #1 (2 planes, 2 cargo, 3 airports)", prob_1))

    # ------------------------------------------------
    # 2) Normal Problem B: (2 cargo, 2 planes, 2 airports)
    #    => (2 + 2)^2 = 4^2=16 cargo placements * 2^2=4 plane placements => 64 total states
    # ------------------------------------------------
    cargos = ["C1", "C2"]
    planes = ["P1", "P2"]
    airports = ["SFO", "JFK"]

    dom_2 = create_air_cargo_domain(cargos, planes, airports)
    init_2 = {
        "At_C1": "JFK",
        "At_C2": "SFO",
        "At_P1": "SFO",
        "At_P2": "SFO",
        "Empty_P1": True,
        "Empty_P2": True
    }
    # The goal: swap cargo positions
    goal_2 = {
        "At_C1": "SFO",
        "At_C2": "JFK"
    }
    prob_2 = Planning_problem(dom_2, init_2, goal_2)
    problems.append(("Normal Problem #2 (2 planes, 2 cargo, 2 airports)", prob_2))

    # ------------------------------------------------
    # 3) Normal Problem C: (3 cargo, 2 planes, 2 airports)
    #    => (2+2=4) => 4^3=64 cargo placements; planes => 2^2=4 => total 64*4=256 states
    # ------------------------------------------------
    cargos = ["C1", "C2", "C3"]
    planes = ["P1", "P2"]
    airports = ["SFO", "JFK"]

    dom_3 = create_air_cargo_domain(cargos, planes, airports)
    init_3 = {
        "At_C1": "SFO",
        "At_C2": "JFK",
        "At_C3": "SFO",
        "At_P1": "SFO",
        "At_P2": "JFK",
        "Empty_P1": True,
        "Empty_P2": True
    }
    goal_3 = {
        "At_C1": "JFK",
        "At_C2": "SFO",
        "At_C3": "JFK"
    }
    prob_3 = Planning_problem(dom_3, init_3, goal_3)
    problems.append(("Normal Problem #3 (2 planes, 3 cargo, 2 airports)", prob_3))

    # ------------------------------------------------
    # 4) Subgoals Problem A: (2 cargo, 2 planes, 3 airports)
    #    Multiple subgoals: e.g. C1->JFK, C2->ORD, and P1->SFO
    # ------------------------------------------------
    cargos = ["C1", "C2"]
    planes = ["P1", "P2"]
    airports = ["SFO", "JFK", "ORD"]

    dom_4 = create_air_cargo_domain(cargos, planes, airports)
    init_4 = {
        "At_C1": "SFO",
        "At_C2": "JFK",
        "At_P1": "ORD",
        "At_P2": "SFO",
        "Empty_P1": True,
        "Empty_P2": True
    }
    goal_4 = {
        "At_C1": "JFK",   # subgoal 1
        "At_C2": "ORD",   # subgoal 2
        "At_P1": "SFO"    # subgoal 3
    }
    prob_4 = Planning_problem(dom_4, init_4, goal_4)
    problems.append(("Subgoals Problem #1 (2 planes, 2 cargo, 3 airports)", prob_4))

    # ------------------------------------------------
    # 5) Subgoals Problem B: (2 cargo, 2 planes, 3 airports)
    #    Another multi-subgoal arrangement
    # ------------------------------------------------
    cargos = ["C1", "C2"]
    planes = ["P1", "P2"]
    airports = ["SFO", "JFK", "ORD"]

    dom_5 = create_air_cargo_domain(cargos, planes, airports)
    init_5 = {
        "At_C1": "ORD",
        "At_C2": "ORD",
        "At_P1": "SFO",
        "At_P2": "JFK",
        "Empty_P1": True,
        "Empty_P2": True
    }
    goal_5 = {
        "At_C1": "SFO",    # subgoal 1
        "At_C2": "JFK",    # subgoal 2
        "At_P2": "ORD"     # subgoal 3 (plane P2 must end up at ORD)
    }
    prob_5 = Planning_problem(dom_5, init_5, goal_5)
    problems.append(("Subgoals Problem #2 (2 planes, 2 cargo, 3 airports)", prob_5))

    # ------------------------------------------------
    # 6) Subgoals Problem C: (2 cargo, 2 planes, 3 airports)
    #    Yet another multi-subgoal scenario
    # ------------------------------------------------
    cargos = ["C1", "C2"]
    planes = ["P1", "P2"]
    airports = ["SFO", "JFK", "ORD"]

    dom_6 = create_air_cargo_domain(cargos, planes, airports)
    init_6 = {
        "At_C1": "SFO",
        "At_C2": "ORD",
        "At_P1": "JFK",
        "At_P2": "JFK",
        "Empty_P1": True,
        "Empty_P2": True
    }
    goal_6 = {
        "At_C1": "ORD",     # subgoal 1
        "At_C2": "SFO",     # subgoal 2
        "At_P1": "ORD",     # subgoal 3
        "At_P2": "SFO"      # subgoal 4
    }
    prob_6 = Planning_problem(dom_6, init_6, goal_6)
    problems.append(("Subgoals Problem #3 (2 planes, 2 cargo, 3 airports)", prob_6))

    ############################################################
    # PROBLEM #7: Using 2 planes, 7 cargos, 4 airports
    # Domain size = 2*(7*2*4) + (2*4*(4-1)) = 112 + 24 = 136 actions
    ############################################################
    planes_7 = ["P1", "P2", "P3"]
    cargos_7 = [f"C{i}" for i in range(1, 8)]   # C1...C7
    airports_7 = ["SFO", "JFK", "ORD", "ATL"]

    domain_7 = create_air_cargo_domain(cargos_7, planes_7, airports_7)

    # Initial state: each plane is at an airport and most cargos are co-located with one plane.
    init_7 = {
        "At_P1": "SFO",
        "At_P2": "JFK",
        "At_P3": "ORD",

        "At_C1": "SFO",
        "At_C2": "JFK",
        "At_C3": "ORD",

        "At_C4": "JFK",
        "At_C5": "ORD",
        "At_C6": "ATL",

        "At_C7": "SFO",
        

        "Empty_P1": True,
        "Empty_P2": True,
        "Empty_P3": True
    }

    # Goal: move cargos to new airports and reposition planes.
    goal_7 = {
        "At_C1": "JFK",
        "At_C2": "ORD",
        "At_C3": "ATL",
        "At_C4": "SFO",
        "At_C5": "JFK",
        "At_C6": "ATL",
        "At_C7": "ATL",
        
        "At_P1": "ATL",
        "At_P2": "ORD",
        "At_P3": "JFK",
    }

    prob_7 = Planning_problem(domain_7, init_7, goal_7)
    problems.append(("Bigger Problem #7 (2 planes, 7 cargo, 4 airports)", prob_7))


    ############################################################
    # PROBLEM #8: Using 2 planes, 5 cargos, 5 airports
    # Domain size = 2*(5*2*5) + (2*5*(5-1)) = 100 + 40 = 140 actions
    ############################################################
    planes_8 = ["P1", "P2"]
    cargos_8 = [f"C{i}" for i in range(1, 7)]   # C1...C6
    airports_8 = ["SFO", "JFK", "ORD", "ATL", "LAX"]

    domain_8 = create_air_cargo_domain(cargos_8, planes_8, airports_8)

    init_8 = {
        "At_P1": "SFO",
        "At_P2": "ORD",

        "At_C1": "SFO",
        "At_C2": "SFO",
        "At_C3": "JFK",
        "At_C4": "ORD",
        "At_C5": "LAX",
        "At_C6": "ATL",
        "Empty_P1": True,
        "Empty_P2": True
    }

    goal_8 = {
        "At_C1": "JFK",
        "At_C2": "ATL",
        "At_C3": "LAX",
        "At_C4": "SFO",
        "At_C5": "ORD",
        "At_C6": "LAX",

        "At_P1": "ATL",
        "At_P2": "JFK",
    }

    prob_8 = Planning_problem(domain_8, init_8, goal_8)
    problems.append(("Bigger Problem #8 (2 planes, 5 cargo, 5 airports)", prob_8))


    ############################################################
    # PROBLEM #9: Using 2 planes, 7 cargos, 4 airports (different initial/goal)
    # Domain size = same as Problem #7: 136 actions
    ############################################################
    planes_9 = ["P1", "P2"]
    cargos_9 = [f"C{i}" for i in range(1, 8)]   # C1...C7
    airports_9 = ["SFO", "JFK", "ORD", "ATL"]

    domain_9 = create_air_cargo_domain(cargos_9, planes_9, airports_9)

    init_9 = {
        "At_P1": "JFK",
        "At_P2": "ORD",
        "At_C1": "JFK",
        "At_C2": "ORD",
        "At_C3": "ORD",
        "At_C4": "ATL",
        "At_C5": "SFO",
        "At_C6": "SFO",
        "At_C7": "SFO",
        "Empty_P1": True,
        "Empty_P2": True    
    }

    goal_9 = {
        "At_C1": "SFO",
        "At_C2": "ATL",
        "At_C3": "JFK",
        "At_C4": "ORD",
        "At_C5": "ATL",
        "At_C6": "JFK",
        "At_C7": "ORD",
        "At_P1": "ATL",
        "At_P2": "SFO",
    }

    prob_9 = Planning_problem(domain_9, init_9, goal_9)
    problems.append(("Bigger Problem #9 (2 planes, 7 cargo, 4 airports; variant)", prob_9))


    return problems



if __name__ == "__main__":
    all_problems = generate_all_problems()
    
    records = []  # we'll store dicts: { 'problem': str, 'domain_size': int, 'time_no_h': float/str, 'time_with_h': float/str }

    for i, (title, prob) in enumerate(all_problems, start=1):
        print(f"\n=== PROBLEM {i}: {title} ===")
        
        domain_size = len(prob.prob_domain.actions)
        print("Number of actions in domain:", domain_size)

        # 1) Solve WITHOUT heuristics
        start_nh = time.time()
        plan_nh = plan_with_timeout(prob, use_heuristics=False, max_time=300)
        end_nh = time.time()

        if plan_nh is None:
            # Distinguish if it's a real "No plan" or a "timeout"?
            # For simplicity, let's call everything "Timeout/No plan" if None.
            # You can add logic in plan_with_timeout to differentiate.
            time_nh_str = "Timeout/No plan"
            print("No plan found or timed out (without heuristics).")
            plan_string = "No plan found"
        else:
            # Found a plan
            time_nh_str = f"{end_nh - start_nh:.8f}"
            print(f"Plan found (without heuristics) in {time_nh_str}s; length = {len(plan_nh)}")
            plan_string = " -> ".join(a.name for a in plan_nh)
            print("Plan:", plan_string)

        # 2) Solve WITH heuristics
        start_wh = time.time()
        plan_wh = plan_with_timeout(prob, use_heuristics=True, max_time=300)
        end_wh = time.time()

        if plan_wh is None:
            time_wh_str = "Timeout/No plan"
            print("No plan found or timed out (with heuristics).")
        else:
            time_wh_str = f"{end_wh - start_wh:.8f}"
            print(f"Plan found (with heuristics) in {time_wh_str}s; length = {len(plan_wh)}")
            plan_string = " -> ".join(a.name for a in plan_wh)
            print("Plan:", plan_string)
        # Collect record
        records.append({
            "problem_name": title,
            "plan_length": len(plan_nh) if plan_nh else "Plan not found",
            "plan_no_heuristic": " -> ".join(a.name for a in plan_nh) if plan_nh else "Plan not found",
            "plan_with_heuristic": " -> ".join(a.name for a in plan_wh) if plan_wh else "Plan not found",
            "domain_actions": domain_size,
            "time_no_heuristic": time_nh_str,
            "time_heuristic": time_wh_str
        })

    # Convert to DataFrame & save to CSV
    df = pd.DataFrame(records)
    df.to_csv("air_cargo_results.csv", index=False)
    print("\nSaved results to air_cargo_results.csv")
    print(df)