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


test_input = """1
2
-3
3
-2
0
4""".splitlines()

data = test_input

def initial_shift(data):
    return v(*list(range(len(data))))

def parse_input(data):
    result = [int(l) for l in data]
    return v(*result)


def build_table(parsed):
    l = len(parsed)
    result = []
    for i, value in enumerate(parsed):
        result.append([(i - 1) % l, (i+1) % l, value])
    return result


def _rotate(current: pvector, index, shift):
    d(f"---")
    d(current, index, shift)
    start = shift[index]
    by = current[start]
    to  = (start + by) % (len(current) - 1)
    if to == 0:
        to = len(current) - 1
    d(f"{by} moves from {start} to {to}")
    if start == to:
        return current, shift
    after = current
    if start < to:
        for i in range(start, to):
            after = after.set(i, current[i+1])
            d(f"shifting {i=}, {shift=}")
            shift = shift.set(shift.index(i+1), i)
        d(f"almost: {shift=}")
        after = after.set(to, by)
        shift = shift.set(index, to)
        d(f"after: {shift=}, {index=}, {to=}")
    else:
        d(f"working from {start=} down to {to=}")
        for i in range(start, to, -1):
            d(f"moving {i-1} ({current[i-1]}) to the right")
            after = after.set(i, current[i-1])
            shift = shift.set(shift.index(shift[i-1]), i)
            d(f"shifting {i=}, {shift=}")
        after = after.set(to, by)
        shift = shift.set(index, to)
            
    d(f"{after=} ({shift=})")
    return after, shift


def print_table(table, level=0):
    if level >= DEBUG:
        s = rowify(table)
        print(", ".join(s))

def rowify(table):
    i = table[0][1]
    s = [str(table[0][2])]
    while i!= 0:
        s.append(str(table[i][2]))
        i = table[i][1]
    return s

def move(table, index, steps):
    count = steps % len(table)
    dir = True
    if count > len(table) // 2:
        dir = False
        count = len(table) - count
    to = index
    for i in range(count):
        to = table[to][dir]
    d(f"Asked to move index {index} ({table[index][2]}) by {steps}. Tried going right {count}. Got {to} ({table[to][2]}).", level=5)
    return to

def rotate(current, index, *args):
    d(f"Starting rotate {index=} - {current[index][2]} - {current=}")
    # print_table(current)
    row = current[index]
    prev, next_, val = row
    l = len(current)
    # d("current row:", prev, next_, val)
    steps = abs(val) % l
    if steps == 0:
        return current
    # d(f"Linking {val} original {index=} to {to=} ({current[to]})")
    to = move(current, index, val)
    current[prev][1] = next_
    current[next_][0] = prev
    
    before = (val < 0)  # If True, insert before else after
    after = not before
    other = current[to][after]
    current[to][after] = index
    current[index][before] = to
    current[index][after] = other
    current[other][before] = index
    # pprint(current)
    # print_table(current)
    # d(f"Returning {current=}")
    return current
    
    # pprint(table)

# assert ==

# data = read_input("20_input.txt")

from tqdm import tqdm

def _rotate_all(vector):
    shift = initial_shift(vector)    
    for i in tqdm(range(len(vector))):
        vector, shift = rotate(vector, i, shift)
    return vector


def _get_numbers(vector):
    start = vector.index(0)
    d(start, level=10)
    s = 0
    # for i in [1, 2, 5]:
    for i in [1000, 2000, 3000]:
        print(i, r:=vector[(start + i) % len(vector)])
        s+= r
    return s

def _test_rotate(parsed):
    init = initial_shift(parsed)
    current, shift = rotate(parsed, 0, init)
    assert current == pvector([2, 1, -3, 3, -2, 0, 4])
    current, shift = rotate(current, 1, shift)
    assert current == pvector([1, -3, 2, 3, -2, 0, 4])
    current, shift = rotate(current, 2, shift)
    assert current == pvector([1, 2, 3, -2, -3, 0, 4])
    current, shift = rotate(current, 3, shift)
    assert current == pvector([1, 2, -2, -3, 0, 3, 4])
    current, shift = rotate(current, 4, shift)
    assert current == pvector([1, 2, -3, 0, 3, 4, -2])
    current, shift = rotate(current, 5, shift)
    assert current == pvector([1, 2, -3, 0, 3, 4, -2])
    current, shift = rotate(current, 6, shift)
    assert current == pvector([1, 2, -3, 4, 0, 3, -2])

def test_build_table(parsed):
    table = build_table(parsed)
    pprint(table)

def test_rotate(table, maxi=1):
    expected = [['1', '-3', '3', '-2', '0', '4', '2'],
 ['1', '-3', '2', '3', '-2', '0', '4'],
 ['1', '2', '3', '-2', '-3', '0', '4'],
 ['1', '2', '-2', '-3', '0', '3', '4'],
 ['1', '2', '-3', '0', '3', '4', '-2'],
 ['1', '2', '-3', '0', '3', '4', '-2']]
    print_table(table)
    results = []
    for i in range(maxi):
        print(f"test rotate({i})")
        table = rotate(table, i)
        print(f"{i} - {rowify(table)} - {expected[i]}")
        assert rowify(table) == expected[i]
        results.append(rowify(table))
    pprint(results)
    assert results==expected
    

def rotate_all(table):
    for i in tqdm(range(len(table))):
        table = rotate(table, i)
    return table

def test_rotate_all(table):
    table = rotate_all(table)
    # pprint(table)
    print(rowify(table))
    expect_table = [[4, 1, 1], [0, 2, 2], [1, 6, -3], [5, 4, 3], [3, 0, -2], [6, 3, 0], [2, 5, 4]]
    expect_row = ['1', '2', '-3', '4', '0', '3', '-2']
    pprint(table)
    pprint(expect_table)
    assert expect_table == table
    assert expect_row == rowify(table)


def get_numbers(table):
    index = [_[2] for _ in table].index(0)
    d(f"Found 0 in row {index}: {table[index]=}")
    results = []
    for i in range(3):
        index = move(table=table, index=index, steps=1000,)
        results.append(table[index][2])
    d(f"Three numbers: {results=}", level=100)
    return sum(results)
    
        

def test_get_numbers(table):
    number = get_numbers(table)
    assert number == 3
    return number

if __name__ == "__main__":
    # data = test_input
    data = read_input("20_input.txt")
    parsed = parse_input(data)
    
    # test_build_table(parsed=parsed)
    table = build_table(parsed=parsed)
    print(data[:5])
    print(parsed[:5])
    print(table[:5])
    print("----", len(data), len(parsed), len(table))
    print(data[-5:])
    print(parsed[-5:])
    print(table[-5:])
    # test_rotate(table, 6)
    # test_rotate_all(table)
    # table = rotate_all(table)
    # print(test_get_numbers(table))
    print(get_numbers(table))
    # current = rotate_all(parsed)
    # print(current)
    # assert current == pvector([1, 2, -3, 4, 0, 3, -2])
    # s = get_numbers(current)    
    # print(s)
    # assert s == 3