from collections import defaultdict
from copy import copy
from copy import deepcopy
from functools import lru_cache
import datetime as dt
import operator as op
from pprint import pprint

from attrs import define
import attrs
# import matplotlib.pyplot as plt
# import networkx as nx
# import numpy as np
# import pandas as pd
import pyrsistent
from pyrsistent import pvector, v

def read_input(s):
    with open(f"./input/{s}") as f:
        data = f.read().splitlines()
        return data

def pw(a, b, op=op.and_):
    return v([op(*_) for _ in zip(a, b)])

DEBUG = 100

def d(*s, level=0):
    if level>=DEBUG:
        print(*s)


test_input = """....#..
..###.#
#...#.#
.#...##
#.###..
##.#.##
.#..#..""".splitlines()

toy_input=""".....
..##.
..#..
.....
..##.
.....""".splitlines()

data = test_input
elves = defaultdict(lambda: None)
directions = {}
global_dir = "N"

def elf_area():
    global elves
    mini = min(_[0] for _ in elves.values())
    maxi = max(_[0] for _ in elves.values())
    minj = min(_[1] for _ in elves.values())
    maxj = max(_[1] for _ in elves.values())
    return (maxi + 1 - mini) * (maxj + 1 - minj) - len(elves)

def parse_input(data):
    global elves
    global directions
    elves = defaultdict(lambda: None)
    directions = {}
    elf_index = 0
    for i, line in enumerate(data):
        for j,c in enumerate(line):
            if c == "#":
                elves[elf_index] = (i,j)
                elf_index += 1
    directions = {_: "N" for _ in elves}
    d(f"Found {len(elves)} elves, covering {elf_area()} ground.", level=100)
    return elves, directions

next = {
    "N": "S",
    "S": "W",
    "W": "E",
    "E": "N",
}

def print_elves():
    mini = min(_[0] for _ in elves.values())
    maxi = max(_[0] for _ in elves.values())
    minj = min(_[1] for _ in elves.values())
    maxj = max(_[1] for _ in elves.values())
    taken_positions = elves.values()
    for i in range(mini, maxi + 1):
        for j in range(minj, maxj + 1):
            if (i,j) in taken_positions:
                print("#", end='')
            else:
                print(".", end='')
        print("")
            

WIND = {
    "N": (-1, 0),
    "S": (1,0),
    "W": (0, -1),
    "E": (0, 1),
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SW": (1, -1),
    "SE": (1, 1),
}

def get_single_proposal(pos, dir):
    global elves
    global global_dir
    i, j = pos
    taken_positions = elves.values()
    # pprint(taken_positions)
    found = ""
    for w, delta in WIND.items():
        if (i+delta[0], j+ delta[1]) in taken_positions:
            found += w
    if found == "":
        return None  # all free
    dir = global_dir
    for _ in range(4):
        if dir not in found:
            return i + WIND[dir][0], j + WIND[dir][1]
        dir = next[dir]
    return None  # blocked                

def get_proposals():
    proposals = {}
    no_proposal = 0
    for i, pos in elves.items():
        proposal = get_single_proposal(pos, directions[i])
        if proposal:
            d(f"Elf {i} wants to go {pos} -> {proposal}", level=-10)
            proposals[i] = proposal
        else:
            no_proposal += 1
    return proposals, no_proposal
                
from collections import Counter
def execute_proposals(proposals):
    global global_dir
    count = Counter(proposals.values())
    free = {k for k,v in count.items() if v == 1}
    # print(free)
    for i, p in proposals.items():
        if p in free:
            elves[i] = p
            # directions[i] = next[directions[i]]
    global_dir = next[global_dir]
    return len(free)

# assert ==

# data = read_input("20_input.txt")
def round():
    proposals, no_proposal = get_proposals()
    moved = execute_proposals(proposals)
    d(f"{no_proposal=}, {moved=}, {elf_area()=}", level=100)
    return moved

if __name__ == "__main__":
    # parse_input(toy_input)
    # data = test_input
    data = read_input("23_input.txt")
    parse_input(data)
    # print(elf_area())
    for k in range(1, 10_000):
        # print("Round", k)
        moved = round()
        # print_elves()
        if k % 12 == 0:
            print(f"We have finished {k} rounds - and {moved} elves still moved.")
        if moved == 0:
            break
    print(f"Finished after {k} rounds")
    
