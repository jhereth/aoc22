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

DEBUG = 20

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
            d(f"{i:4d}: line {l}", level=10)
            s = l.split(": ")
            result[s[0]] = s[1]
    return result


def solve(yells, name, start):
    line = yells[name]
    d(f"processing {line=}")
    try:
        result = int(line)
        return result, True
    except:
        pass
    for operation in ["+", "-", "*", "/"]:
        if operation in line: break
    d(f"Found {operation} in {line}")
    split = line.split(" " + operation + " ")
    d(f"Looking for lhs rhs in {split}")
    lhs, rhs = split[0], split[1]
    if lhs == "humn":
        r, _ = solve(yells, rhs, None)
        return r, False
    if rhs == "humn":
        r, _ = solve(yells, lhs, None)
        return r, False 
    # print(f"Trying to solve {line}")
    lr, lhuman = solve(yells, lhs, start)
    rr, rhuman = solve(yells, rhs, start)
    if lhuman or rhuman:
        return (lr, lhuman, rr, rhuman), False
    result = eval(str(lr) + operation + str(rr))
    return result, True
        
# assert ==
from fractions import Fraction
from decimal import Decimal
# @lru_cache
def reduce(yells, name, final=None, level=0):
    line = yells[name]
    # if final:
    #     if abs(final - int(final)) > 0.1:
            # raise Exception(f"Deviation in final: {name=} - {final=} - {line=}")
    ops = {
        "+":op.add,
        "-": op.sub,
        "*": op.mul,
        "/": op.truediv,
    }
    opp = {
        "+": op.sub,
        "-": op.add,
        "*": op.truediv,
        "/": op.mul, 
    }
    d(f"reducing {name=} || {line=} ||  {final=} || {level=}", level=100)
    if name == "hmn":
        raise Exception("WTF4")
    try:
        # result = Fraction(int(line), 1)
        result = Decimal(int(line))
        d(f"returning {result} for || {name=} || {line=} ||  {final=} || {level=}", level=100)
        return result, False 
    except:
        pass
    for operation in ["+", "-", "*", "/"]:
        if operation in line: break
    else:
        raise Exception(f"No known operation in {line=} ({name=})")
    d(f"found operation {operation} in {line}", level=-10)
    split = line.split(" " + operation + " ")
    lhs, rhs = split[0], split[1]
    if lhs == "humn":
        if final is None:
            d(f"returning 'Human found' for || {name}: {line} || {lhs=} || {rhs=} || {line=} {final=} || {level=}", level=100)
            return None, True
        rr, rhuman = reduce(yells, rhs, None, level=level+1)
        d(f"left Human Found almost a Solution {final=} - || {line=} || {operation=} - {opp[operation]=} - {rr=} - {rhuman=}", level=100)
        equation = str(final) + str(opp[operation]) + str(rr)
        result = opp[operation](final, rr)
        # if abs(result - int(result)) > 0.1:
        #     raise Exception(f"Deviation: {result=} - {equation=} - {name=} - {final=} - {rr=}")
        if name == "root":
            d(f"Found root Solution {result=} || {final=} - {operation=} - {opp[operation]=} - {rr=} - {rhuman=}", level=100)
            raise Exception(f"{line} with rr={rr}")
        d(f"Found final Solution {result=} || {final=} - {operation=} - {opp[operation]=} - {rr=} - {rhuman=}", level=100)
        raise Exception(f"The human variable is {result} ({line=} - {name=} - {final=} - {rr=})") 
    if rhs == "humn":
        if final is None:
            d(f"returning 'Human found' for || {name:} {line} ||  {lhs=} | {rhs=} | {final=} || {level=}", level=100)
            return None, True
        lr, _ = reduce(yells, lhs, final, level=level+1)
        d(f"right Human Found almost a Solution {final=} - || {line=} || {operation=} - {ops[operation]=} - {lr=} - {lhuman=}", level=100)
        result = ops[operation](lr, final)
        # if abs(result - int(result)) > 0.1:
        #     raise Exception(f"Deviation: {result=} - {equation=} - {name=} - {final=} - {lr=}")
        raise Exception(f"The human variable is {result=} ({line=} - {name=} - {final=} - {lr=})") 
    d(f"No human found in {final}=={line} (yet).", level=-5)
    lr, lhuman = reduce(yells, lhs, None, level=level+1)
    rr, rhuman = reduce(yells, rhs, None, level=level+1)
    if lhuman and rhuman:
        raise Exception("WTF")
    if lhuman:
        side = lhs
        val_other_side = rr
        # new_final = rr
        equation = str(final) + str(opp[operation]) + str(rr)
        if final is not None:
            new_final = opp[operation](final, rr)
            d(f"{name} left Human: Calculated {new_final=} from {line=} and {final=} - {rr=}", level=30)
    if rhuman:
        side = rhs
        val_other_side = lr
        if final is not None:
            if operation in ["-", "/"]:
                new_final = ops[operation](lr ,final)
            else:
                new_final = opp[operation](final, lr)
            d(f"{name} right Human: Calculated {new_final=} from {line=} and {final=} - {lr=}", level=30)
    if lhuman or rhuman:
        if final is None:
            if name != "root":
                d(f"returning summary 'Human found' for || {name}: {line} || {lhuman=} || {rhuman=} || {line=} ||  {final=} || {level=}", level=100)
                return None, True
            d(f"Descending to solve: {name=} | {line=} | {val_other_side=}")
            val, human = reduce(yells, side, val_other_side, level=level+100_000)
            raise Exception(f"WTF2 - {val} - {human}")
        d(f"Trying to solve with {new_final=} | {name=} | {line=}")
        val, human = reduce(yells, side, new_final, level=level+1)
        raise Exception("WTF3")
    # result = eval(str(lr) + operation + str(rr))
    result = ops[operation](lr, rr)
    # if abs(result - int(result)) > 0.1:
        # raise Exception(f"Deviation: {result=} - {equation=} - {name=} - {final=} - {lr=}")
    d(f"returning {result} for || {name=} || {line=} ||  {final=} || {level=}", level=100)
    return result, False

if __name__ == "__main__":
    data="""root: this / number
number: 7
this: foo / humn
foo: 7""".splitlines()
    # pprint(data)
    data = test_input
    # data = read_input("21_input.txt")
    parsed = parse_input(data)
    # pprint(parsed)
    # print(solve(parsed, "humn") )
    # print(solve(parsed, "ptdq") )
    # print(solve(parsed, "root") )  # 364367103397416
    print(reduce(parsed, "root", None))  # 3782852515583
    
