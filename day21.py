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

DEBUG = 0

def d(*s, level=0):
    if level>=DEBUG:
        print(*s)


test_input = """root: pppw + sjmn
dbpl: 5
cczh: sllz + lgvd
zczc: 2
ptdq: humn - dvpt
dvpt: 3
lfqf: 4
humn: 5
ljgn: 2
sjmn: drzm * dbpl
sllz: 4
pppw: cczh / lfqf
lgvd: ljgn * ptdq
drzm: hmdt - zczc
hmdt: 32
""".splitlines()


def parse_input(data):
    result = {}
    for i, l in enumerate(data):
        if l:
            d(f"{i:4d}: line {l}")
            s = l.split(": ")
            result[s[0]] = s[1]
    return result


def solve(yells, name, start):
    line = yells[name]
    d(f"processing {line}")
    try:
        result = int(line)
        return result
    except:
        pass
    for operation in ["+", "-", "*", "/"]:
        if operation in line: break
    d(f"Found {operation} in {line}")
    split = line.split(" " + operation + " ")
    d(f"Looking for lhs rhs in {split}")
    lhs, rhs = split[0], split[1]
    if lhs == "humn":
        r = solve(yells, rhs, None)
        raise Exception(f"Human found staring from {start} ({line} - {r})")
    if rhs == "humn":
        r = solve(yells, lhs, None)
        raise Exception(f"Human found staring from {start} ({line} {r})")
    lr = solve(yells, lhs, start)
    rr = solve(yells, rhs, start)
    result = eval(str(lr) + operation + str(rr))
    return result
        
# assert ==



if __name__ == "__main__":
    data = test_input
    data = read_input("21_input.txt")
    parsed = parse_input(data)
    # print(solve(parsed, "humn") )
    # print(solve(parsed, "ptdq") )
    # print(solve(parsed, "root") )  # 364367103397416
    
    # part 2
    # root: wvbw + czwc
    # print(solve(parsed, "wvbw", "wvbw"))  # Human found staring from wvbw (humn - qflr - 594.0)
    print(solve(parsed, "czwc", "czwc")) # 594
    print("^^ czwc")
    print(solve(parsed, "qflr", "qflr"))
    print("^^ qflr")


