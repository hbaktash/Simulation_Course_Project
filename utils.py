import random
import math


def generate_exponential(param):
    u = random.uniform(0, 1)
    return (-1/param)*math.log(u)


def generate_bernouli(param: float):
    if random.uniform(0, 1) < param:
        return 1
    else:
        return 0


def generate_type(type_param):
    if generate_bernouli(type_param) == 1:
        return 1
    else:
        return 2
