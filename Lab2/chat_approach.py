from aircargo import Strips, STRIPS_domain, Planning_problem

def create_air_cargo_domain(cargos, planes, airports):
    """
    Tworzy domenę STRIPS dla prostego problemu 'air cargo',
    w którym mamy zbiór ładunków (cargos), samolotów (planes)
    i lotnisk (airports).
    """
    # 1) Definiujemy słownik (feature_domain_dict) który mapuje nazwę cechy (feature)
    #    na zbiór możliwych wartości dla tej cechy.
    #    W tym podejściu cecha "At_X" oznacza "Gdzie jest X?", a wartości to nazwy lotnisk
    #    lub samolotów (jeśli X to ładunek).
    
    feature_domain_dict = {}
    
    # Dla każdego ładunku definiujemy cechę "At_C" (gdzie jest cargo?),
    # której możliwe wartości to wszystkie lotniska oraz wszystkie samoloty.
    for c in cargos:
        feature_domain_dict[f"At_{c}"] = set(airports) | set(planes)
    
    # Dla każdego samolotu definiujemy cechę "At_P", której możliwe wartości to lotniska.
    # Zakładamy, że samolot nie "wchodzi" do innego samolotu, więc wartości to tylko lotniska.
    for p in planes:
        feature_domain_dict[f"At_{p}"] = set(airports)
    
    # 2) Definiujemy zbiór akcji (actions) w ujęciu STRIPS:
    #    - LOAD (załaduj ładunek na samolot)
    #    - UNLOAD (rozładuj ładunek z samolotu)
    #    - FLY (poleć samolotem z jednego lotniska na drugie)

    actions = set()

    # 2A) LOAD: 
    #   Parametry: cargo c, plane p, airport a
    #   Prekondycje:
    #     - ładunek c jest w miejscu a (At_{c} = a)
    #     - samolot p jest w tym samym miejscu a (At_{p} = a)
    #   Efekty:
    #     - ładunek c jest teraz w samolocie p (At_{c} = p)
    #     - tym samym ładunek nie jest już na lotnisku a
    #
    #   Tworzymy osobny obiekt Strips dla każdej kombinacji c, p, a.
    for c in cargos:
        for p in planes:
            for a in airports:
                name = f"LOAD_{c}_onto_{p}_at_{a}"
                preconds = {
                    f"At_{c}": a,  # ładunek c jest na lotnisku a
                    f"At_{p}": a   # samolot p jest również na lotnisku a
                }
                effects = {
                    f"At_{c}": p   # w efekcie ładunek c będzie w samolocie p
                }
                actions.add(Strips(name, preconds, effects))

    # 2B) UNLOAD:
    #   Parametry: cargo c, plane p, airport a
    #   Prekondycje:
    #     - ładunek c jest aktualnie w samolocie p (At_{c} = p)
    #     - samolot p jest na lotnisku a (At_{p} = a)
    #   Efekty:
    #     - ładunek c jest teraz na lotnisku a (At_{c} = a)
    #
    #   Podobnie jak wyżej – generujemy akcje dla wszystkich możliwych kombinacji.
    for c in cargos:
        for p in planes:
            for a in airports:
                name = f"UNLOAD_{c}_from_{p}_at_{a}"
                preconds = {
                    f"At_{c}": p,  # ładunek c jest w samolocie p
                    f"At_{p}": a   # samolot p jest na lotnisku a
                }
                effects = {
                    f"At_{c}": a   # ładunek c trafia na lotnisko a
                }
                actions.add(Strips(name, preconds, effects))

    # 2C) FLY:
    #   Parametry: plane p, from, to
    #   Prekondycje:
    #     - samolot p znajduje się na lotnisku "from" (At_{p} = from)
    #   Efekty:
    #     - samolot p znajduje się teraz na lotnisku "to" (At_{p} = to)
    #   Oczywiście "from" != "to", więc nie chcemy generować akcji FLY p z lotniska do tego samego lotniska.
    
    
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

    # 3) Tworzymy i zwracamy obiekt STRIPS_domain z zdefiniowanym słownikiem cech i zbiorem akcji
    return STRIPS_domain(feature_domain_dict, actions)


def goal_satisfied(state, goal):
    return all(state.get(k) == v for k, v in goal.items())

def apply_action(state, action):
    # Check if the action's preconditions hold
    if not all(state.get(k) == v for k, v in action.preconds.items()):
        return None
    new_state = state.copy()
    # Apply the action's effects
    for k, v in action.effects.items():
        new_state[k] = v
    return new_state

def plan(problem: Planning_problem):
    from collections import deque
    start = problem.initial_state
    frontier = deque([(start, [])])
    visited = set()
    while frontier:
        state, path = frontier.popleft()
        state_key = tuple(sorted(state.items()))
        if state_key in visited:
            continue
        visited.add(state_key)
        if goal_satisfied(state, problem.goal):
            return path
        for action in problem.prob_domain.actions:
            new_state = apply_action(state, action)
            if new_state is not None:
                frontier.append((new_state, path + [action]))
    return None


# Poniżej przykładowy problem planistyczny "na próbę",
# w którym mamy 1 ładunek (C1), 1 samolot (P1) i 2 lotniska (SFO, JFK).
# Cel: przetransportować ładunek C1 z lotniska SFO na lotnisko JFK.

if __name__ == "__main__":
    # 1) Definiujemy listy ładunków, samolotów i lotnisk
    cargos = ["C1", "C2"]
    planes = ["P1", "P2"]
    airports = ["SFO", "JFK"]

    # 2) Tworzymy domenę
    air_cargo_domain = create_air_cargo_domain(cargos, planes, airports)

    # 3) Określamy stan początkowy
    #    - ładunek C1 jest na lotnisku SFO
    #    - samolot P1 jest na lotnisku SFO
    initial_state = {
        "At_C1": "SFO",
        "At_P1": "SFO",
        "At_P2": "SFO",
        "At_C2": "JFK"
    }

    # 4) Określamy cel: ładunek C1 ma być na lotnisku JFK
    goal = {
        "At_C1": "JFK", "At_C2": "SFO", "At_P1": "SFO", "At_P2": "SFO"
    }

    # 5) Tworzymy konkretny problem planistyczny
    air_cargo_problem = Planning_problem(air_cargo_domain, initial_state, goal)

    # Na tym etapie air_cargo_problem można przekazać do algorytmu planującego
    # (np. wyszukiwanie w przestrzeni stanów) i znajdować plan z LOAD, FLY, UNLOAD.
    print("Domena (cechy i akcje):")
    print("Features:", air_cargo_domain.feature_domain_dict)
    print("Actions:", air_cargo_domain.actions)
    print("\nProblem:\n", air_cargo_problem)

    # Get plan
    solution = plan(air_cargo_problem)
    if solution:
        print("\nPlan found:")
        for step in solution:
            print(step.name)
    else:
        print("\nNo plan found.")