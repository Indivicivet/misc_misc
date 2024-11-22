from dataclasses import dataclass, field
import random
from typing import Self


@dataclass
class GraphNode:
    children: list[Self] = field(default_factory=list)
    tag: int = -1

    def __post_init__(self):
        self.tag = random.randint(1, 1_000_000_000)

    def __hash__(self):
        return self.tag

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.tag == other.tag
        return NotImplemented

    def __repr__(self):
        return f"{type(self).__name__}(tag={str(self.tag)})"


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
    curr.children.append(init)
    return total


@dataclass
class GraphSearchResult:
    search_history: list[GraphNode] = field(default_factory=list)
    future_queue: list[GraphNode] = field(default_factory=list)
    found: bool = False
    searched_set: set[GraphNode] = field(default_factory=set)  # for speed

    @classmethod
    def depth_first(cls, structure, function, target):
        res = cls()
        res.future_queue.append(next(iter(structure.items)))
        while res.future_queue:
            curr = res.future_queue.pop()
            res.search_history.append(curr)
            res.future_queue.extend(
                [x for x in curr.children if x not in res.searched_set]
            )
            res.searched_set.add(curr)
            if function[curr] == target:
                res.found = True
                break
        return res


if __name__ == "__main__":
    c3 = get_cyclic(3)
    f_vals = {}
    a_node = next(iter(c3.items))
    i = 0
    f_vals[a_node] = i
    curr_node = a_node.children[0]
    while curr_node != a_node:
        i += 1
        f_vals[curr_node] = i
        curr_node = curr_node.children[0]
    print(f_vals)
    print(GraphSearchResult.depth_first(c3, f_vals, 2))
    print(GraphSearchResult.depth_first(c3, f_vals, 4))
