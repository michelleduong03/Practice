# Given an array of integers, return the number of pairs of adjacent elements that have opposite parity (one even, one odd).

# Example:
# [1, 4, 7, 2, 5] → 3

nums = [1, 4, 7, 2, 5]

def count_pairs(nums):
    count = 0

    for i in range(len(nums) - 1):
        if nums[i] % 2 != nums[i+1] % 2:
            count += 1

    return count

nums = [2,2,3,3,3,4]
def freq(nums):
    seen = {}
    max = 0
    res = 0
    for num in nums:
        if num not in seen:
            seen[num] = 1
        else:
            seen[num] += 1
    for k,v in enumerate(seen):
        print(k)
        print(v)
        if v > max:
            max = v
            res = k

    return res

print(freq(nums))

# PYTHON PATTERNS

#     # Hash map counting
# arr = [1, 2, 3]
# counts = {}
# for x in arr:
#     counts[x] = counts.get(x, 0) + 1

#     # 2D grid movement template
# rows, cols = len(grid), len(grid[0])

# for r in range(rows):
#     for c in range(cols):
#         # visit grid[r][c]

#     # BFS template
# from collections import deque

# def bfs(grid, sr, sc):
#     q = deque([(sr, sc)])
#     visited = set([(sr, sc)])

#     while q:
#         r, c = q.popleft()

#         for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
#             nr, nc = r+dr, c+dc

#             if 0 <= nr < rows and 0 <= nc < cols:
#                 if (nr, nc) not in visited:
#                     visited.add((nr, nc))
#                     q.append((nr, nc))




# How many groups of 1's?
grid = [
 [1,1,0],
 [0,1,0],
 [1,0,1]
] # Answer → 3 groups

    # Look at first point

    # If it's 1

    # Explore neighbors

    # Stop when hit 0

    # Count that as one island

    # PSEUDOCODE:
    # for every cell in grid:
    #     if cell is 1 and not visited:
    #         BFS from here
    #         islands += 1

# I’ll iterate the grid. Whenever I see an unvisited 1, 
# I’ll run BFS to mark the whole connected component, 
# then increment my island counter.

# if diagonals count too, we count the additional for directions
# (1,1), (1,-1), (-1,1), (-1,-1) (added in for loop)

# ↖  ↑  ↗
# ←  X  →
# ↙  ↓  ↘

from collections import deque

def countIslands(grid):
    rows = len(grid)
    cols = len(grid[0])
    visited = set()
    islands = 0

    def bfs(r, c):
        q = deque([(r, c)])
        visited.add((r, c))

        while q:
            r, c = q.popleft()

            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                nr, nc = r + dr, c + dc

                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr][nc] == 1 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and (r, c) not in visited:
                bfs(r, c)
                islands += 1

    return islands
