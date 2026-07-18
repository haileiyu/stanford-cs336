import heapq
from collections import Counter
from functools import total_ordering

@total_ordering
class Entry:
    """Heap entry: higher count wins; ties broken by lexicographically larger item."""
    __slots__ = ("count", "item")

    def __init__(self, count, item):
        self.count = count
        self.item = item

    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count   # higher count surfaces first
        return self.item > other.item         # then lexicographically larger

    def __eq__(self, other):
        return self.count == other.count and self.item == other.item

    def __repr__(self):
        return f"Entry({self.item!r}, count={self.count})"


class FreqTracker:
    def __init__(self):
        self.counts = Counter()
        self.heap = []

    def add(self, item):
        self.counts[item] += 1
        heapq.heappush(self.heap, Entry(self.counts[item], item))

    def most_frequent(self):
        while self.heap:
            top = self.heap[0]
            if top.count == self.counts[top.item]:
                return top.item, top.count
            heapq.heappop(self.heap)
        return None


if __name__ == "__main__":
    # tie-breaking demo: all counts equal, largest item wins
    h = []
    heapq.heappush(h, Entry(2, "apple"))
    heapq.heappush(h, Entry(2, "zebra"))
    heapq.heappush(h, Entry(2, "abb"))
    heapq.heappush(h, Entry(2, "mango"))
    print(h[0])                # Entry('zebra', count=2)

    # # tracker demo
    # t = FreqTracker()
    # for x in ["apple", "banana", "apple", "banana"]:
    #     t.add(x)
    # print(t.most_frequent())   # ('banana', 2)