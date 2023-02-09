from math import log


class Node(object):
    def __init__(self, key, val):
        self.degree = 0
        self.key = key
        self.val = val
        self.p = None
        self.child = None
        self.left = None
        self.right = None
        self.mark = False


class FibonacciHeap(object):
    def __init__(self):
        self.n = 0
        self.root = None
        self.max = None

    @staticmethod
    def make_heap():
        return FibonacciHeap()

    def insert(self, x, y=None):
        if not isinstance(x, Node):
            x = Node(x, y)

        x.left = x.right = x

        if self.max is None:
            self.root = x
            self.max = x
        else:

            x.right = self.root.right
            x.left = self.root
            self.root.right.left = x
            self.root.right = x

            if x.key > self.max.key:
                self.max = x

        self.n = self.n + 1

    def find_max(self):
        return self.max

    def extract_max(self):
        z = self.max

        if z is not None:
            if z.child:
                children = [c for c in self.__iter_nodes(z.child)]

                for x in children:
                    if self.root is None:
                        self.root = x
                    else:
                        x.right = self.root.right
                        x.left = self.root
                        self.root.right.left = x
                        self.root.right = x

                    x.p = None

            if z is self.root:
                self.root = z.right

            z.left.right = z.right
            z.right.left = z.left

            if z == z.right:
                self.max = None
                self.max = None
                self.root = None
            else:
                self.max = z.right
                self.__consolidate()

            self.n -= 1

        return z

    def union(self, heap):
        return self.__add__(heap)

    def increase_key(self, x, k):
        if k < x.key:
            raise ValueError('New key is smaller than current key')

        x.key = k
        y = x.p

        if y is not None and x.key > y.key:
            self.__cut(x, y)
            self.__cascading_cut(y)

        if x.key > self.max.key:
            self.max = x

    def delete(self, x):
        self.increase_key(x, float('inf'))
        self.extract_max()

    def __cut(self, x, y):
        if y.child == y.child.right:
            y.child = None
        elif y.child == x:
            y.child = x.right
            x.right.p = y

        x.left.right = x.right
        x.right.left = x.left
        y.degree -= 1

        if self.root is None:
            self.root = x
        else:
            x.right = self.root.right
            x.left = self.root
            self.root.right.left = x
            self.root.right = x

        x.p = None
        x.mark = False

    def __cascading_cut(self, y):
        z = y.p

        if z is not None:
            if not y.mark:
                y.mark = True
            else:
                self.__cut(y, z)
                self.__cascading_cut(z)

    def __consolidate(self):
        a = [None] * int(log(self.n) * 2)
        nodes = [n for n in self.__iter_nodes(self.root)]

        for x in nodes:
            d = x.degree

            while a[d] is not None:
                y = a[d]

                if x.key < y.key:
                    temp = x
                    x, y = y, temp

                self.__heap_link(y, x)
                a[d] = None
                d += 1

            a[d] = x

        self.max = None

        for i in range(int(log(self.n) * 2)):
            if a[i] is not None:
                if self.max is None or a[i].key > self.max.key:
                    self.max = a[i]

    @staticmethod
    def __iter_nodes(head, reverse=False):
        node = halt = head
        f = False

        while True:
            if node == halt and f:
                break
            elif node == halt:
                f = True

            yield node

            if reverse:
                node = node.left
            else:
                node = node.right

    def __heap_link(self, y, x):
        if y is self.root:
            self.root = y.right

        y.left.right = y.right
        y.right.left = y.left
        y.left = y.right = y

        if x.child is None:
            x.child = y
        else:
            y.right = x.child.right
            y.left = x.child
            x.child.right.left = y
            x.child.right = y

        x.degree += 1
        y.p = x
        y.mark = False

    def __add__(self, heap):
        if not isinstance(heap, FibonacciHeap):
            raise ValueError('Heap is not a Fibonacci Heap')

        h = self.make_heap()
        h.max = self.max
        h.root = self.root

        tmp = heap.root.left
        heap.root.left = h.root.left
        h.root.left.right = heap.root
        h.root.left = tmp
        h.root.left.right = h.root

        if self.max is None or (heap.max is not None and heap.max.key > self.max.key):
            h.max = heap.max

        h.n = self.n + heap.n
        return h

    def __len__(self) -> int:
        return self.n
