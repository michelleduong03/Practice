# 🚀 Capital One CodeSignal OA Prep Guide (70 min)

## Format Overview

* 4 questions / 70 minutes
* All DSA (no React/UI)
* Typical structure
  **Q1 Easy → Q2 Easy+ → Q3 Matrix BFS → Q4 Optimization**

### Best Order

👉 **1 → 2 → 4 → 3**

---

# Python Essentials

```python
from collections import defaultdict, deque
```

### Common Conversions

```python
int("42")
s.split("/")
" ".join(words)
```

---

# Q1 – Simulation / Hashmap

### Bank System Template

```python
def solve(ops):
    user = {}

    for op in ops:
        if op[1] not in user:
            user[op[1]] = 0

        if op[0] == "CREATE":
            user[op[1]] = 0

        elif op[0] == "DEPOSIT":
            user[op[1]] += int(op[2])

        elif op[0] == "TRANSFER":
            s,r,a = op[1], op[2], int(op[3])
            if user[s] >= a:
                user[s] -= a
                user[r] += a

    return user
```

---

# Q2 – Intervals

```python
intervals = [[1,2], [3,6], [4, 8], [8, 12]]
def merge(intervals):
    intervals.sort()
    res = [intervals[0]]

    for s,e in intervals[1:]:
        if s <= res[-1][1]: # end of last merged interval
            res[-1][1] = max(res[-1][1], e)
        else:
            res.append([s,e])
    return res
```

---

# Q3 – MATRIX BFS (CORE)

```python
from collections import deque
dirs = [(1,0),(-1,0),(0,1),(0,-1)]

def largest_island(grid):
    rows, cols = len(grid), len(grid[0])
    visited = set()

    def bfs(r,c):
        q = deque([(r,c)])
        visited.add((r,c))
        size = 1

        while q:
            x,y = q.popleft()
            for dx,dy in dirs:
                nx,ny = x+dx, y+dy
                if (0<=nx<rows and 0<=ny<cols
                    and (nx,ny) not in visited
                    and grid[nx][ny]==1):

                    visited.add((nx,ny))
                    q.append((nx,ny))
                    size += 1
        return size

    best = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]==1 and (r,c) not in visited:
                best = max(best, bfs(r,c))
    return best
```

---

# Q4 – Optimization

### Prefix Sum

```python
def prefix(nums):
    pre=[0]
    for n in nums:
        pre.append(pre[-1]+n)
    return pre
```

### Histogram (monotonic stack)

```python
def largestRectangle(h):
    stack=[]; best=0
    h.append(0)

    for i,x in enumerate(h):
        while stack and h[stack[-1]]>x:
            height=h[stack.pop()]
            left=stack[-1] if stack else -1
            best=max(best, height*(i-left-1))
        stack.append(i)

    return best
```

---

# 🧠 BFS Flashcards

**When use BFS?**
Grid, shortest path, connected components.

**Visited needed?**
YES – prevents infinite loops.

**Queue stores?**
Coordinates to explore next.

**4-direction check**

```python
0<=nx<rows and 0<=ny<cols
```

---

# ✈️ Plane Drills (Do These)

Got it — you want **starter code for each of these 10 drills**, just like CodeSignal would give: function signature + place to write logic + sample test.

Below is copy-paste ready Python.

---

# ✈️ Plane Drills – STARTER CODE

## 1. Frequency Count

```python
def solution(words):
    # return dictionary of counts
    counts = {}

    # TODO
    for word in words:
        if word not in counts:
            counts[word] = 1
        else:
            counts[word] += 1

    return counts


# test
print(solution(["a","b","a"]))   # {"a":2,"b":1}
```

---

## 2. Simplify Path

```python
def solution(path):
    # return simplified unix path
    # hint: use stack

    # TODO

    return ""


print(solution("/a/../b//c/"))   # "/b/c"
```

---

## 3. Number of Islands (BFS)

```python
from collections import deque

def solution(grid):
    # return number of islands of 1s

    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    visited = set()

    def bfs(r,c):
        # TODO
        pass

    # TODO loop grid

    return 0
```

---

## 4. Merge Intervals

```python
def solution(intervals):
    intervals.sort()
    result = []

    # TODO

    return result


print(solution([[1,3],[2,5],[8,10]]))
```

---

## 5. Prefix Sum Range

```python
def build_prefix(nums):
    prefix = [0]
    # TODO
    return prefix


def range_sum(prefix, L, R):
    # return sum nums[L:R+1] in O(1)
    # TODO
    return 0
```

---

## 6. Rotate Matrix 90°

```python
def solution(matrix):
    # rotate in-place or return new

    # TODO

    return matrix
```

---

## 7. Candy Crush Gravity

```python
def solution(board):
    # drop non-zero cells down each column

    # TODO

    return board
```

---

## 8. Meeting Rooms II

```python
def solution(intervals):
    # return min number of rooms needed

    # hint: sort start/end separately

    # TODO

    return 0
```

---

## 9. Histogram Largest Rectangle

```python
def solution(heights):
    stack = []
    best = 0

    heights.append(0)

    # TODO monotonic stack

    return best
```

---

## 10. LRU Cache

```python
class LRU:
    def __init__(self, capacity):
        # TODO
        pass

    def get(self, key):
        # TODO
        pass

    def put(self, key, value):
        # TODO
        pass
```

---

# How to Use on the Plane

For each problem:

1. Read starter
2. Implement
3. Test with example
4. Move on — 15 min max per problem

---

# 🕒 Time Plan

* Q1 – 8 min
* Q2 – 10 min
* Q4 – 20 min
* Q3 – rest

---

# CodeSignal Rules

DO

* brute → then optimize
* clear names
* handle empty input

DON’T

* one-line hero code
* spend 40 min on Q3

---

## 70-MIN MOCK (try offline)

**Q1** – Bank simulation
**Q2** – Merge intervals
**Q3** – Largest island BFS
**Q4** – Histogram


```python
Q3 – BFS / Island Variant

Grid example:

grid = [
 [1,1,0,0],
 [1,0,0,2],
 [0,0,1,1],
 [2,0,1,1]
]


1 = land

0 = water

2 = treasure

Return a matrix where each land cell has the distance to nearest treasure.

from collections import deque

def treasure_distance(grid):
    if not grid or not grid[0]:
        return []

    rows, cols = len(grid), len(grid[0])
    dist = [[-1] * cols for _ in range(rows)]  # initialize distances
    visited = set()
    q = deque()

    # Step 1: enqueue all treasures
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                q.append((r, c, 0))  # (row, col, distance)
                visited.add((r, c))
                dist[r][c] = 0  # treasure distance = 0

    # Step 2: BFS
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    while q:
        r, c, d = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # check boundaries
            if 0 <= nr < rows and 0 <= nc < cols:
                # only visit land (1) and not visited
                if grid[nr][nc] == 1 and (nr, nc) not in visited:
                    dist[nr][nc] = d + 1
                    visited.add((nr, nc))
                    q.append((nr, nc, d + 1))

    return dist

```