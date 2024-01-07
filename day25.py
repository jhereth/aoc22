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


test_input = """1=-0-2
12111
2=0=
21
2=01
111
20012
112
1=-1=
1-12
12
1=
122""".splitlines()

toy_input="""1121-1110-1=0""".splitlines()

data = test_input

def parse_input(data):
    result = []
    for line in data:
        result.append(snafu_to_int(line))
    return result


SNAFU = {
    "2": 2,
    "1": 1,
    "0": 0,
    "-": -1,
    "=": -2,
}

UFANS = {
    d: s for (s,d) in SNAFU.items()
}

def snafu_to_int(snafu: str) -> int:
    s = 0
    while snafu:
        s *= 5
        c, snafu = snafu[0], snafu[1:]
        s += SNAFU[c]
    return s



def convert(d: int) -> str:
    snafu = ""
    carry = 0
    while d:
        d += carry
        carry = 0
        d, r = d//5, d % 5
        if r > 2:
            carry = 1
            r -=5
        snafu = UFANS[r] + snafu
    if carry:
        snafu = "1" + snafu
    return snafu
        

def sum_and_convert(lines):
    s = sum(lines)
    return convert(s)

   


if __name__ == "__main__":
    # data = toy_input
    data = test_input
    data = read_input("25   _input.txt")
    parsed = parse_input(data)
    result = sum_and_convert(parsed)
    # for d in [2022, 12345, 314159265]:
        # print(d, convert(d))
    print(result) 
