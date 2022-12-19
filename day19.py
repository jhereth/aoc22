from collections import defaultdict
from copy import copy
from copy import deepcopy
from functools import lru_cache
import datetime as dt
import operator as op
from pprint import pprint

from attrs import define
import attrs
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyrsistent
from pyrsistent import pvector as V

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


test_input = """Blueprint 1: Each ore robot costs 4 ore.  Each clay robot costs 2 ore.  Each obsidian robot costs 3 ore and 14 clay.  Each geode robot costs 2 ore and 7 obsidian.
Blueprint 2: Each ore robot costs 2 ore.  Each clay robot costs 3 ore.  Each obsidian robot costs 3 ore and 8 clay.  Each geode robot costs 3 ore and 12 obsidian.  """.splitlines()

data = test_input

@define(frozen=True)
class Blueprint:
    identifier: int
    ore_cost: int
    clay_cost: int
    obsidian_ore_cost: int
    obsidian_clay_cost: int
    geode_ore_cost: int
    geode_obsidian_cost: int



def parse_input(data):
    blueprints = []
    for line in data:
        stripped = [_.strip(".:") for _ in line.split()]
        # d(stripped)
        numbers = [int(_) for _ in stripped if _.isnumeric()]
        # d(numbers)
        bp = Blueprint(*numbers)
        # d(bp)
        blueprints.append(bp)
    return blueprints

blueprints = parse_input(data)
d(blueprints, level=50)
bp = blueprints[1]


@define(frozen=True)
class Supplies:
    ore_bots: int = 1
    clay_bots: int = 0
    obsidian_bots: int = 0
    geode_bots: int = 0
    ore: int = 0
    clay: int = 0
    obsidian: int = 0
    cracked_geods : int = 0


from enum import Enum
R = Enum("RobotTypes", ["Ore", "Clay", "Obsidian", "Geode"])

CACHE = {
    "processed": set(),
    "max_found": 0,
    "best_path": (),
}

def find_decisions(bp: Blueprint, supplies: Supplies):
    d("XXY 01", supplies)
    if (
        (supplies.obsidian >= bp.geode_obsidian_cost) 
        and (supplies.ore >= bp.geode_ore_cost)
    ):
        d(f"XXY 02: build Geode. {supplies.obsidian=}, {supplies.ore=} >= {bp.geode_obsidian_cost=},{bp.geode_ore_cost=}")
        yield R.Geode
    else:
        d("XXY 03: no geode")
        if (
                supplies.clay >= bp.obsidian_clay_cost
            and supplies.ore >= bp.obsidian_ore_cost
            and supplies.obsidian_bots < bp.geode_obsidian_cost
        ):
            yield R.Obsidian
        if (
            supplies.ore >= bp.clay_cost
            and supplies.clay_bots < bp.obsidian_clay_cost
        ):
                    yield R.Clay
        if  (
                supplies.ore >= bp.ore_cost
                and supplies.ore_bots < max(bp.ore_cost, bp.clay_cost, bp.obsidian_ore_cost, bp.geode_ore_cost)
        ):                
            yield R.Ore
        yield None

def optimize(time_remaining: int, supplies: Supplies, bp: Blueprint, path=()):
    global CACHE
    d("XXX 10", time_remaining, path, "max_found:", CACHE["max_found"], supplies)
    max_found = CACHE["max_found"]
    # d(max_found, CACHE["max_found"], len(CACHE["processed"]))
    if (t:=(time_remaining, supplies, bp)) in CACHE["processed"]:
        d("XXX 11", t, path)
        return max_found
    else:
        # d("XXX 12: continue with", t)
        # pd(CACHE["processed"])
        CACHE["processed"].add(t)
        # d("XX", len(CACHE["processed"]))
        # pd(CACHE["processed"])

    new_supplies = attrs.asdict(supplies)
    new_supplies["ore"] += supplies.ore_bots
    new_supplies["clay"] += supplies.clay_bots
    new_supplies["obsidian"] += supplies.obsidian_bots
    new_supplies["cracked_geods"] += supplies.geode_bots
    d(f"XXX 12: {new_supplies=} {supplies=}")
    if time_remaining <= 1:
        new_max = new_supplies["cracked_geods"]
        d(f"XXX 13: {new_max=} {path=}")
        if new_max > max_found:
            CACHE["max_found"] = new_max
            CACHE["best_path"] = path
            d(f"{dt.datetime.now().strftime('%HH:%MM:%SS')} - {time_remaining=} {new_max=} {max_found=} len={len(CACHE['processed'])} {CACHE['max_found']=} {supplies=}")
            return new_max
        return max_found
            # raise ValueError(f"{time_remaining} is too low ({bp=}, {supplies=})")
    if (achievable_upper_bound:=new_supplies["cracked_geods"] + new_supplies["geode_bots"]*(time_remaining) + ((time_remaining) * (time_remaining + 1) // 2)) <= max_found:
        d(f"XXX 18 achievable stop: {achievable_upper_bound=} {max_found=}")
        d(f"XXX 19 {new_supplies['cracked_geods']=}, {new_supplies['geode_bots']=}, {time_remaining=}, {(time_remaining) * (time_remaining - 1) // 2}, {max_found}")
        return max_found
    branches = {}
    for dec in find_decisions(bp, supplies):
        dec_supplies = copy(new_supplies)
        dec_path = path + (dec,)
        d("XXX 20 predec:", dec,  dec_path, path, dec_supplies)
        # d("XXX3", time_remaining, dec, dec_path)
        if dec == R.Geode:
                dec_supplies["ore"] -= bp.geode_ore_cost
                dec_supplies["obsidian"] -= bp.geode_obsidian_cost
                dec_supplies["geode_bots"] += 1
        elif dec == R.Obsidian:
                dec_supplies["ore"] -= bp.obsidian_ore_cost
                dec_supplies["clay"] -= bp.obsidian_clay_cost
                dec_supplies["obsidian_bots"] +=1
        elif dec ==  R.Clay:
                dec_supplies["ore"] -= bp.clay_cost
                dec_supplies["clay_bots"] += 1
        elif dec ==  R.Ore:
                dec_supplies["ore"] -= bp.ore_cost
                dec_supplies["ore_bots"] += 1 
        d("XXX 29 postdec:", dec_supplies, dec_path)
        dec_max = optimize(time_remaining-1, Supplies(**dec_supplies), bp, path=dec_path)
        branches[dec] = dec_max
    d(f"XXX 30: postdec. {path=}")
    best = max(branches, key=branches.get)
    new_max = branches[best]
    d(f"XXX 90 - {len(path)} - {path+(best,)}: {new_max=}")
    # pd(branches)
    if new_max > max_found:
        d(f"XXX 99: {dt.datetime.now().strftime('%H:%M:%S')} - {time_remaining=} {new_max=} {max_found=} len={len(CACHE['processed'])} {CACHE['max_found']=} {dec_path=} {supplies=}", level=100)
        CACHE["max_found"] = new_max
        return  branches[best]
    return max_found




supplies = Supplies()
# supplies20 = Supplies(ore_bots=1, clay_bots=4, obsidian_bots=2, ore=1, clay=21, obsidian=5)

supp = {}
day = 0

supp[0] = Supplies()

supp[5] = Supplies(
    ore_bots=1,
    clay_bots=1,
    obsidian_bots=0,
    geode_bots=0,
    ore=2,
    clay=1,
    obsidian=0,
    cracked_geods=0,    
)
supp[10] = Supplies(
    ore_bots=1,
    clay_bots=3,
    obsidian_bots=0,
    geode_bots=0,
    ore=3,
    clay=12,
    obsidian=0,
    cracked_geods=0,    
)
supp[11] = Supplies(
    ore_bots=1,
    clay_bots=3,
    obsidian_bots=0,
    geode_bots=0,
    ore=4,
    clay=15,
    obsidian=0,
    cracked_geods=0,    
)
supp[12] = Supplies(
    ore_bots=1,
    clay_bots=3,
    obsidian_bots=1,
    geode_bots=0,
    ore=2,
    clay=4,
    obsidian=0,
    cracked_geods=0,    
)
supp[13] = Supplies(
    ore_bots=1,
    clay_bots=4,
    obsidian_bots=1,
    geode_bots=0,
    ore=1,
    clay=7,
    obsidian=1,
    cracked_geods=0,    
)
supp[14] = Supplies(
    ore_bots=1,
    clay_bots=4,
    obsidian_bots=1,
    geode_bots=0,
    ore=2,
    clay=11,
    obsidian=2,
    cracked_geods=0,    
)
supp[15] = Supplies(
    ore_bots=1,
    clay_bots=4,
    obsidian_bots=1,
    geode_bots=0,
    ore=3,
    clay=15,
    obsidian=3,
    cracked_geods=0,
)
supp[16] = Supplies(
    ore_bots=1,
    clay_bots=4,
    obsidian_bots=2,
    geode_bots=0,
    ore=1,
    clay=5,
    obsidian=4,
    cracked_geods=0,
)
supp[17] = Supplies(
    ore_bots=1,
    clay_bots=4,
    obsidian_bots=2,
    geode_bots=0,
    ore=2,
    clay=9,
    obsidian=6,
    cracked_geods=0,
)
supp[18] = Supplies(
    ore_bots=1,
    clay_bots=4,
    obsidian_bots=2,
    geode_bots=0,
    ore=3,
    clay=13,
    obsidian=8,
    cracked_geods=0,
)
supp[19] = Supplies(
    ore_bots=1,
    clay_bots=4,
    obsidian_bots=2,
    geode_bots=1,
    ore=2,
    clay=17,
    obsidian=3,
    cracked_geods=1,
)
supp[20] = Supplies(
    ore_bots=1,
    clay_bots=4,
    obsidian_bots=2,
    geode_bots=1,
    ore=3,
    clay=21,
    obsidian=5,
    cracked_geods=1,
)
supp[21] = Supplies(
    ore_bots=1,
    clay_bots=4,
    obsidian_bots=2,
    geode_bots=1,
    ore=4,
    clay=25,
    obsidian=7,
    cracked_geods=2,
)
supp[22]= Supplies(
    ore_bots=1,
    clay_bots=4,
    obsidian_bots=2,
    geode_bots=2,
    ore=3,
    clay=29,
    obsidian=2,
    cracked_geods=3,
)
supp[23] = Supplies(
    ore_bots=1,
    clay_bots=4,
    obsidian_bots=2,
    geode_bots=2,
    ore=4,
    clay=33,
    obsidian=4,
    cracked_geods=5,
)
supp[24] = Supplies(
    ore_bots=1,
    clay_bots=4,
    obsidian_bots=2,
    geode_bots=2,
    ore=5,
    clay=37,
    obsidian=6,
    cracked_geods=7,
)

# print(optimize(24-day, supp[day], bp, ()))

# for day in range(10, 25):
#     x = optimize(24-day, supp[day], bp)
#     print(f"Checking day {day} - {x=}")
#     assert  x == 9
print("Starting...")
def sum_quality(blueprints: list[Blueprint]):
    global CACHE
    s = 0
    for bp  in blueprints:
        CACHE = {
            "processed": set(),
            "max_found": 0,
            "best_path": (),
        }
        print(f"Processing {bp.identifier}")
        supplies = Supplies()
        quality = optimize(24, supplies=supplies, bp=bp, path=())
        s += bp.identifier * quality
        print(f"We found quality {quality} for Blueprint {bp.identifier}. (Cache-Size {len(CACHE['processed'])}")
    print(f"Qualitysum is {s}")
    return s

data = read_input("19_input.txt")
blueprints = parse_input(data)
# pprint(blueprints)
# sum_quality(blueprints=blueprints)

def big(blueprints):
    levels = []
    for i in range(3):
        CACHE = {
            "processed": set(),
            "max_found": 0,
            "best_path": (),
        }
        bp=blueprints[i]
        print(f"Processing Blueprint {bp.identifier}")
        pprint(bp)
        supplies = Supplies()
        start_time = dt.datetime.now()
        geodes = optimize(time_remaining=32,
                          supplies=supplies,
                          bp=bp,
                          path=(),
                          )
        end_time = dt.datetime.now()
        levels.append(geodes)
        print(f"We can open {geodes} geodes with blueprint {bp.identifier}.")
        print(f"processed >{len(CACHE['processed'])} entries in {end_time - start_time}")
    print(f"Final Result: *{levels}={levels[0] * levels[1] * levels[2]}.")
        
big(blueprints=blueprints)
print(len(CACHE["processed"]))