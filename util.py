import numpy as np
from functools import reduce
from math import *
import itertools

def flatten(img):
    return np.reshape(img, img.size, order='C')

def inflate(x, shape):
    return np.reshape(x, shape)

def rotate(xs, k):
    return xs[-k:] + xs[:-k]

def product(xs):
    return reduce(lambda x, y: x * y, xs)

def concat(xss):
    return sum(xss, [])

def powerset(xs, keep):
    return [sorted(it) for it in itertools.chain.from_iterable([itertools.combinations(xs,n) for n in range(len(xs)+1)]) if all([item in it for item in keep])]

def dfs(graph, start, goal, path=[], visited=None):
    if path is None:
        path = []
    if visited is None:
        visited = {start}
    if goal(start):
        yield path
    for (edge, node) in graph[start]:
        if not node in visited:
            yield from dfs(graph, node, goal, path + [edge], visited | {node})

def bfs(graph, start, goal):
    queue = [(start, {start}, [])]
    while queue:
        (vertex, visited, path) = queue.pop(0)
        for (edge, node) in graph[vertex]:
            if node not in visited:
                if goal(node):
                    yield path + [edge]
                else:
                    queue.append((node, visited | {node}, path + [edge]))
