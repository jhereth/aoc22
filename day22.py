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

DEBUG = -20

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
    return field, len(data), max(len(_) for _ in data)

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
            raise ValueError(f"parse_direction: something strange happened at position {i=}, {row=}")
    result.append((M.Move, int(steps)))
    return result
            
def get_start_pos(field):
    data = field[0]
    for i in range(1, 1+field[2]):
        # print(data[1])
        if data[1].get(i):
            return (1, i, "R")


wrap_map = {}
wrap_map[(12, 16)] = {
    (0, 0, "R"): (0,0, "R"),
    (0, 0, "D"): (0,0, "D"),
    (0, 0, "L"): (0,0, "L"),
    (0, 0, "U"): (0,0, "U"), 
}
wrap_map[(200, 150)] = {
    (0, 0, "R"): (0,0, "R"),
    (0, 0, "D"): (0,0, "D"),
    (0, 0, "L"): (0,0, "L"),
    (0, 0, "U"): (0,0, "U"), 
}
wrap_map[(4, 4)] = {
    (0, 2, "R"): (2,3, "L"),
    (0, 2, "D"): (1,2, "D"),
    (0, 2, "L"): (1,1, "D"),
    (0, 2, "U"): (1,0, "D"), 
    (1, 0, "R"): (1,1, "R"),
    (1, 0, "D"): (2,2, "U"),
    (1, 0, "L"): (2,3, "U"),
    (1, 0, "U"): (0,2, "D"), 
    (1, 1, "R"): (1,2, "R"),
    (1, 1, "D"): (2,2, "R"),
    (1, 1, "L"): (1,0, "L"),
    (1, 1, "U"): (0,2, "R"), 
    (1, 2, "R"): (2,3, "D"),
    (1, 2, "D"): (2,2, "D"),
    (1, 2, "L"): (1,1, "L"),
    (1, 2, "U"): (0,2, "U"), 
    (2, 2, "R"): (2,3, "R"),
    (2, 2, "D"): (1,0, "U"),
    (2, 2, "L"): (1,1, "U"),
    (2, 2, "U"): (1,2, "U"), 
    (2, 3, "R"): (0,2, "L"),
    (2, 3, "D"): (1,0, "R"),
    (2, 3, "L"): (2,2, "L"),
    (2, 3, "U"): (1,2, "L"), 
}
wrap_map[(50, 50)] = {
    (0, 1, "R"): (0, 2, "R"),
    (0, 1, "D"): (1, 1, "D"),
    (0, 1, "L"): (2, 0, "R"),
    (0, 1, "U"): (3, 0, "R"), 
    (0, 2, "R"): (2,1, "L"),
    (0, 2, "D"): (1,1, "L"),
    (0, 2, "L"): (0,1, "L"),
    (0, 2, "U"): (3,0, "U"), 
    (1, 1, "R"): (0,2, "U"),
    (1, 1, "D"): (2,1, "D"),
    (1, 1, "L"): (2,0, "D"),
    (1, 1, "U"): (0,1, "U"), 
    (2, 0, "R"): (2,1, "R"),
    (2, 0, "D"): (3,0, "D"),
    (2, 0, "L"): (0,1, "R"),
    (2, 0, "U"): (1,1, "R"), 
    (2, 1, "R"): (0,2, "L"),
    (2, 1, "D"): (3,0, "L"),
    (2, 1, "L"): (2,0, "L"),
    (2, 1, "U"): (1,1, "U"), 
    (3, 0, "R"): (2,1, "U"),
    (3, 0, "D"): (0,2, "D"),
    (3, 0, "L"): (0,1, "D"),
    (3, 0, "U"): (2,0, "U"), 
}



def jump(i, j, face, cube_sides):
    cube_h, cube_w = cube_sides
    d(f"jump: {i=}, {j=}, {face=} with cube sides {cube_w} - {cube_h}", level=30)
    if face == "R":
        j -= 1
    if face == "D":
        i -= 1
    if face == "L":
        j += 1
    if face == "U":
        i += 1
    this_i, this_j = ((i-1) // cube_h), ((j-1) // cube_w)
    d(f"jump: We are on side {this_i}, {this_j}, facing {face}", level = 40)
    next_i, next_j, next_face = wrap_map[cube_sides][this_i, this_j, face]
    d(f"jump: Going to side {next_i=}, {next_j=}, {next_face=}", level=30)
    new_pos = [0,0]
    if next_face in "RL":
        fix_dim = 1
    else:
        fix_dim = 0
    if next_face in "RD":
        fix_val = 0
    else:
        if next_face == "L":
            fix_val = cube_w - 1
        else: # U
            fix_val = cube_h - 1
    new_pos[fix_dim] = fix_val
    var_dim = 1 - fix_dim
    if face in "RL":
        this_var_dim = 0
    else:
        this_var_dim = 1
    if ((face in "RU") == (next_face in "RU")):
        d(f"{face} and {next_face} share sign", level=25)
        var_val = (([i, j][this_var_dim] - 1) % cube_w) # this could be wrong if cube_w != cube_h
        d(f"Calculated {var_val=}", level=25)
    else:
        d(f"{face} and {next_face} NOT sharing sign", level=25)
        var_val = cube_w - 1 - (([i, j][this_var_dim] - 1)  % cube_w) # this could be wrong. But this only happens in Part 2 where cube_h==cube_w
        d(f"Calculated {var_val=}", level=25)
    new_pos[var_dim] = var_val
    new_pos[0] += next_i * cube_h + 1
    new_pos[1] += next_j * cube_w + 1
    d(f"jump: returning", new_pos[0], new_pos[1], next_face )
    return new_pos[0], new_pos[1], next_face 

@lru_cache
def pre_step(i, j, face, cube_sides=(16, 12)):
    _, fh, fw = field
    cube_h, cube_w = cube_sides
    d(f"pre_step: deriving {cube_w=}, {cube_h=} from {cube_sides=}", level=-30)
    if face == "R":
        j+=1
    elif face == "D":
        i+=1
    elif face == "L":
        j-=1
    elif face == "U":
        i-=1
    if ( ((face == "U") and (i % cube_h == 0))
        or ((face == "L") and (j % cube_w == 0))
        or ((face == "D") and (i % cube_h == 1))
        or ((face == "R") and (j % cube_w == 1))
    ):
        d(f"pre_step: jumping {i=}, {j=}, {face=}, {cube_h=}, {cube_w=}, {i % cube_h=}, {j % cube_w=}")
        i, j, face = jump(i, j, face, cube_sides) 
    return i, j, face


@lru_cache
def wrap_step(i,j, face, cube_length=(16,12)):
    fdata, _, _ = field
    i, j, face = pre_step(i, j, face, cube_length)
    while fdata[i].get(j) is None:
        d(f"wrap_step: Moving forward due to None ({i=}, {j=})", level=-5)
        i, j, face = pre_step(i, j, face, cube_length)
    d(f"wrap_step: Returning {i=}, {j=}, {face=}, Value: {fdata[i].get(j)}")
    return i, j, face

@lru_cache
def step(i, j, face, cube_length=(12,12)):
    d(f"step: from {i},{j} to {face} ({cube_length=})", level=-1)
    orig_i, orig_j, orig_face = i, j, face
    fdata, fh, fw = field
    d(f"{fw=}, {fh=}, {fdata[i]}", level=-30)
    i, j, face = wrap_step(i, j, face, cube_length)
    if fdata[i].get(j) == ".": return i, j, face
    if fdata[i].get(j) == "#": return orig_i, orig_j, orig_face


def move(start_pos, direction, cube_length=None):
    if cube_length is None:
        cube_length = field[1:3]
    d(f"Moving from {start_pos} with {direction} ({cube_length=})", level=30)
    i, j, face = start_pos
    dir_type, val = direction
    if dir_type == M.Move:
        for _ in range(val):
            d(f"step {_} of {val} for move {direction} from {start_pos} ({i=}, {j=})", level=25)
            i, j, face = step(i, j, face, cube_length)
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
def moves(pos, moves, cube_length=None):
    for dir in tqdm(moves):
        pos = move(pos, dir, cube_length=cube_length)
    return pos

# assert ==

if __name__ == "__main__":
    # data = test_input
    # dirs, field = parse_input(data)
    # start_pos1 = get_start_pos(field)
    # print(start_pos1)
    # pos1 = moves(start_pos1, dirs)

    # data = read_input("22_input.txt")
    # dirs, field = parse_input(data)
    # # pprint(field)
    # start_pos = get_start_pos(field)
    # print(start_pos)
    # pos = moves(start_pos, dirs)
    # print(pos)  # (149, 34, 'L') -> 1000 * 149 + 4 * 34 + 2
    # print(field[1], field[2])
    # print(start_pos1) # (1, 9, 'R')
    # print(pos1)  #  (6, 8, 'R')
    # data = test_input
    # dirs, field = parse_input(data)
    # print("field dims:", field[1:3])
    # start_pos21 = get_start_pos(field)
    # print(start_pos21)
    # pos21 = moves(start_pos21, dirs, cube_length=(4,4))
    # print(pos21)
    data = read_input("22_input.txt")
    dirs, field = parse_input(data)
    start_pos22 = get_start_pos(field)
    print(start_pos22)
    pos22 = moves(start_pos22, dirs, cube_length=(50,50)) # (153, 50, 'U') -> 1000 * 153 + 4 * 50 + 3  == 153203
    print(pos22)
