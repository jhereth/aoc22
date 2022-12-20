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

DEBUG = 500

def d(*s, level=0):
    if level>=DEBUG:
        print(*s)


test_input = """""".splitlines()

data = test_input

def parse_input(data):
    result = []
    return result



# assert ==

# data = read_input("20_input.txt")

if __name__ == "__main__":
    # parsed = parse_input(data)
    # pprint(parsed)
    print("hello")

