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

DEBUG = 15

def d(*s, level=0):
    if level>=DEBUG:
        print(*s)

blizzards = {}
width = 0
height = 0
cycle_time = 0

toy_input= """#.#####
#.....#
#>....#
#.....#
#...v.#
#.....#
#####.#""".splitlines()

test_input = """#.######
#>>.<^<#
#.<..<<#
#>v.><>#
#<^v^^>#
######.#""".splitlines()

data = test_input

from math import lcm

def parse_input(data):
    global blizzards
    global height
    global width
    global cycle_time
    height = len(data) - 2
    width = len(data[0]) - 2
    blizzards = {}
    minute = 0
    blizzards = defaultdict(set)
    for i, line in enumerate(data):
        for j, c in enumerate(line):
            if c == "#":
                if (
                     (j==0) or (j==width)
                     or (i==0) or (i==height)
                     ):
                    continue
            elif c in ">v<^":
                blizzards[c].add((i-1,j-1))
            elif c == ".":
                continue
            else:
                raise ValueError(f"Strange Thing at {i=}, {j=}")
    # Sanity Checks
    assert data[0][1] == "."
    assert data[height + 1][width] == ".", f"Expect . but {height=}, {width=}, {data[height+1][width]=}, {data[height-1]=}"
    for i in range(1, 1 + height):
        assert data[i][1] in "<>."
        assert data[i][width] in "<>.", f"Expect <>., but {i=}, {width=}, {height=}, {data[i][width ]=}"
    cycle_time = lcm(width, height)
    print(f"Width: {width}, Height: {height}, cycle time: {cycle_time}")


deltas = {
        ">": (0, 1),
        "v": (1, 0),
        "<": (0, -1),
        "^": (-1, 0),
    }

def taken_positions(minute):
    global blizzards
    global height
    global width
    
    taken = defaultdict(list)
    for c in ">v<^":
        for b in blizzards[c]:
            taken[
                ((b[0] + minute * deltas[c][0]) % height, (b[1] + minute * deltas[c][1]) % width)
            ] += c  # last c wins, not counting
    return taken


def find_neighbours(pos, taken):
    global height
    global width
    result = []
    d(f"{pos=} - {taken=}")
    if pos not in taken:
        result.append(pos)
    for c in ">v<^":
        p = (i:=(pos[0] + deltas[c][0]), j:=(pos[1] + deltas[c][1]))
        d(f"Testing {p=}")
        if (
            p not in taken
        and (0 <= i < height) 
        and (0 <= j < width)
        ):
            d(f"Found new candidate {p}")
            result.append(p)
    return result



def print_grid(taken, cands, level=0):
    global height
    global width
    if level < DEBUG:
        return
    for i in range(height):
        for j in range(width):
            if (t:=(i,j)) in taken:
                if len(c:=taken[t]) > 1:
                    print(len(c), end="")
                else:
                    print(c[0], end="")
            elif t in cands:
                print("E", end="")
            else:
                print(".", end="")
        print()

import networkx as nx
from tqdm import tqdm
def build_graph(start_min=1, end_min=5, start_top=False):
    global blizzards
    global height
    global width
    global cycle_time
    G = nx.DiGraph()
    if start_top:
        start_pos = (-1, 0)
        start_neighbour_pos = (0,0)
        goal_pos = (height, width - 1)
        goal_neighbour_pos = (height - 1, width - 1)
    else:
        start_pos = (height, width - 1)
        start_neighbour_pos = (height - 1, width - 1)
        goal_pos = (-1, 0)
        goal_neighbour_pos = (0,0)
    start_node = (*start_pos, start_min - 1)
    goal_node = (*goal_pos, end_min)
    d(f"build_graph: {start_node=}, {goal_node=}", level=10)
    this_cand = {start_pos}
    for minute in tqdm(range(start_min, end_min + 1)):
        next_round = set()
        d(f"We have {len(this_cand)} candidates entering minute {minute=}", level=20)
        d(f"Candidates: {this_cand}", level=10)
        taken = taken_positions(minute=minute)
        for tc in this_cand:
            new_cands = find_neighbours(tc, taken)
            d(f"Found {new_cands=} for {tc=}")
            if new_cands:
                next_round |= set(new_cands)
                for nc in new_cands:
                    G.add_edge((*tc, minute -1), (*nc, minute))
        if goal_neighbour_pos in this_cand:
            d(f"guild_graph: goal connection found", level=10)
            G.add_edge((*goal_neighbour_pos, minute - 1), goal_node)
        this_cand = next_round
        print_grid(taken, this_cand, level=10)
    return G
        


if __name__ == "__main__":
    data = toy_input
    data = test_input
    # data = read_input("24_input.txt")
    parse_input(data)
    print(f"{height=}, {width=}, {len(blizzards[0])=}")
    pprint(blizzards)

    # start_minute = 1
    # max_minute=30
    # start_pos = (-1,0)
    # goal_pos = (height, width -1)
    # G = build_graph(start_min=start_minute, end_min=max_minute, start_top=True)
    # path = nx.shortest_path(G, source=(*start_pos, start_minute - 1), target=(*goal_pos, max_minute))
    # pprint(path)
    # print(f"{len(path) - 1} steps")  
    # Testinput: 18 steps (18==30)
    
    # Round 2
    # start_minute = 18
    # max_minute=50
    # G = build_graph(start_min=start_minute, end_min=max_minute, start_top=False)
    # start_node = (height, width-1, start_minute)
    # goal_node = (-1, 0, max_minute)
    # path = nx.shortest_path(G, source=start_node, target=goal_node)
    # pprint(path)
    # print(f"{len(path) - 1} steps")  
    # Testinput: 23 steps (step 41==50)

    # Round 3
    # start_minute = 41
    # max_minute= 60
    # G = build_graph(start_min=start_minute, end_min=max_minute, start_top=True)
    # start_node = (-1, 0, start_minute)
    # goal_node = (height, width-1, max_minute)
    # path = nx.shortest_path(G, source=start_node, target=goal_node)
    # pprint(path)
    # print(f"{len(path) - 1} steps")  # Testinput: 13 steps (step 54==60) <- 54 is the solution
