import networkx as nx
import random

def independent_cascade(G, seeds, p=0.1, max_steps=100):
    activated = set(seeds)
    newly_activated = set(seeds)
    
    for step in range(max_steps):
        if not newly_activated:
            break
        current_new = set()
        for node in newly_activated:
            for neighbor in G.neighbors(node):
                if neighbor not in activated and random.random() < p:
                    current_new.add(neighbor)
        newly_activated = current_new
        activated.update(current_new)
    
    return activated
