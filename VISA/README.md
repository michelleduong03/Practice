# Visa CodeSignal OA Prep – Michelle Edition

Goal: be comfortable with the exact patterns Visa tests  
Format: 4 questions, ~70 minutes, easy → medium → optimization

Target strategy:
Q1: 100%  
Q2: 100%  
Q3: mostly correct (matrix/graph)  
Q4: smart attempt > brute force

---

## 1. Python Templates You Must Know

### Adjacent Pair Loop
for i in range(len(nums) - 1):
    a = nums[i]
    b = nums[i+1]

### Hashmap Counting
count = {}
for x in nums:
    count[x] = count.get(x, 0) + 1

### 2D Grid Traversal
rows, cols = len(grid), len(grid[0])

for r in range(rows):
    for c in range(cols):
        cell = grid[r][c]

### BFS Template (most important)

from collections import deque

def bfs(grid, sr, sc):
    rows, cols = len(grid), len(grid[0])
    q = deque([(sr, sc)])
    visited = {(sr, sc)}

    while q:
        r, c = q.popleft()

        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc

            if 0 <= nr < rows and 0 <= nc < cols:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))

---

## 2. Patterns Visa Loves

1. Adjacent comparisons  
2. Frequency with hashmaps  
3. Matrix traversal  
4. Connected components  
5. Simple optimization / greedy  
6. Simulation of steps

---

## 3. Practice Problems (Do on Plane)

### P1 – Adjacent Parity
Input: [1,4,7,2,5]  
Count how many neighboring pairs have opposite parity.

```python
def 

```

---

### P2 – Most Frequent
Return element with highest frequency.  
Tie → return smaller.

[2,2,3,3,3,4] → 3

---

### P3 – Count Islands (4-direction)

grid = [
 [1,1,0],
 [0,1,0],
 [1,0,1]
]

Answer → 3

---

### P4 – Matrix Path

Given grid of 0 (free) and 1 (blocked),  
can you reach bottom-right from top-left?

Return True/False using BFS.

---

### P5 – Optimization

Prices = [5,3,4,2,6]  
Find maximum profit from one buy then one sell.

---

## 4. My Game Plan During Test

1. Read ALL questions first (2 min)
2. Do easiest fully
3. Don’t get stuck >12 min on one
4. Use print debugging
5. Submit even partial for Q4

---

## 5. Common Traps

- Off-by-one in loops  
- Forget visited set  
- Not checking bounds in grid  
- Brute force too slow  
- Using global variables

---

## 6. Michelle Cheat Sheet

Parity check:
a % 2 != b % 2

Opposite sign:
a * b < 0

Neighbors 4-dir:
[(1,0),(-1,0),(0,1),(0,-1)]

Neighbors 8-dir:
add (1,1),(1,-1),(-1,1),(-1,-1)

---

## 7. Confidence Rules

- I don’t need perfect 600  
- Clear logic > fancy code  
- BFS template solves most Q3  
- Hashmap solves most Q2

I’ve got this 💪
