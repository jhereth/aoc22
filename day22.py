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

DEBUG = 30

def d(*s, level=0):
    if level>=DEBUG:
        print(*s)


test_input = """        ...#
        .#..
        #...
        ....
...#.......#
........#...
..#....#....
..........#.
        ...#....
        .....#..
        .#......
        ......#.

10R5L5R10L4R5L5""".splitlines()

data = test_input


def parse_field(data):
    field = defaultdict(lambda: defaultdict())
    for i, row in enumerate(data, start=1):
        for j, c in enumerate(row, start=1):
            if c in ".#":
                field[i][j] = c
    return field, len(data), len(data[0])

def parse_input(data):
    dir_row = data.pop()
    # print(dir_row)
    dirs = parse_directions(dir_row)
    # pprint(dirs)
    data.pop()
    field = parse_field(data)
    # pprint(field)
    result = []
    return dirs, field

from enum import Enum

class Dir(Enum):
    R: 0
    D: 1
    L: 2
    U: 3

class M(Enum):
    Move = 0
    Rotate = 1
        
def parse_directions(row):
    result = []
    steps = ""
    for i,c in enumerate(row):
        if c in "RDLU":
            result.append((M.Move, int(steps)))
            steps = ""
            result.append((M.Rotate, c))
        elif c in "0123456789":
            steps += c
        else:
            raise ValueError(f"something strange happened at position {i}")
    result.append((M.Move, int(steps)))
    return result
            
def get_start_pos(field):
    data = field[0]
    for i in range(1, 1+field[2]):
        # print(data[1])
        if data[1].get(i):
            return (1, i, "R")



@lru_cache
def pre_step(i, j, face, cube_length):
    _, fh, fw = field
    if face == "R":
        j+=1
    elif face == "D":
        i+=1
    elif face == "L":
        j-=1
    elif face == "U":
        i-=1    
    if i==0: i=fh
    if j==0: j=fw
    if i>fh: i=1
    if j> fw: j=1
    return i, j


@lru_cache
def wrap_step(i,j, face, cube_length=None):
    fdata, fh, fw = field
    if cube_length is None:
        cube_length = fw
    i, j = pre_step(i, j, face, cube_length)
    while fdata[i].get(j) is None:
        i, j = pre_step(i, j, face, cube_length)
    return i, j

@lru_cache
def step(i, j, face):
    d(f"Step from {i},{j} to {face}", level=-1)
    orig_i, orig_j = i, j
    fdata, fh, fw = field
    d(f"{fw=}, {fh=}, {fdata[i]}", level=-20)
    i, j = wrap_step(i, j, face)
    if fdata[i].get(j) == ".": return i, j
    if fdata[i].get(j) == "#": return orig_i, orig_j


def move(start_pos, direction):
    d(f"Moving from {start_pos} with {direction}", level=70)
    i, j, face = start_pos
    dir_type, val = direction
    if dir_type == M.Move:
        for _ in range(val):
            d(f"step {_} of {val} for move {direction} from {start_pos}", level=-5)
            i, j = step(i, j, face)
        return (i, j, face)
    else: # Rotate
        rots = {
            "R": {
                "R": "D",
                "D": "L",
                "L": "U",
                "U": "R",
            },
            "L":{
                "R": "U",
                "D": "R",
                "L": "D",
                "U": "L",
            }
        }
        return (i, j, rots[val][face])

from tqdm import tqdm
def moves(pos, moves):
    for dir in tqdm(moves):
        pos = move(pos, dir)
    return pos

# assert ==

if __name__ == "__main__":
    data = test_input
    data = read_input("22_input.txt")
    dirs, field = parse_input(data)
    # pprint(field)
    start_pos = get_start_pos(field)
    print(start_pos)
    pos = moves(start_pos, dirs)
    print(pos)  # (149, 34, 'L') -> 1000 * 149 + 4 * 34 + 2
    print(field[1], field[2])

