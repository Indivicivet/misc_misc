from dataclasses import dataclass, field
import random
from typing import Self


@dataclass
class GraphNode:
    children: list[Self] = field(default_factory=list)

    def __init__(self):
        self.tag = random.randint(1_000_000_000)

    def __hash__(self):
        return self.tag

@dataclass
class GraphTotal:
    items: set[GraphNode] = field(default_factory=set)


def get_cyclic(n):
    init = GraphNode()
    curr = init
    total = GraphTotal({init})
    for _ in range(n - 1):
        prev = curr
        curr = GraphNode()
        prev.children.append(curr)
        total.items.add(curr)
    init.children.append(curr)


if __name__ == "__main__":
    print(get_cyclic(3))
