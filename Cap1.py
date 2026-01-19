operations = [
 ["CREATE", "alice"],
 ["DEPOSIT", "alice", "100"],
 ["DEPOSIT", "alice", "50"],
 ["CREATE", "bob"],
 ["DEPOSIT", "bob", "20"],
 ["TRANSFER", "alice", "bob", "70"]
]

def solve(operations):
    user = {}

    for op in operations:

        if op[0] == "CREATE":
            name = op[1]
            user[name] = 0

        elif op[0] == "DEPOSIT":
            name = op[1]
            amount = int(op[2])
            user[name] += amount

        elif op[0] == "TRANSFER":
            sender = op[1]
            receiver = op[2]
            amount = int(op[3])

            if sender in user and receiver in user:
                if user[sender] >= amount:
                    user[sender] -= amount
                    user[receiver] += amount

    return user


print(solve(operations))


grid = [
 [1,1,0,0],
 [1,0,0,1],
 [0,0,1,1],
 [0,1,1,1]
]

# from collections import deque

# dirs = [(1,0),(-1,0),(0,1),(0,-1)]

# def bfs(r, c):
#     q = deque([(r,c)])
#     visited.add((r,c))
#     size = 1

#     while q:
#         x,y = q.popleft()

#         for dx,dy in dirs:
#             nx, ny = x+dx, y+dy

#             if (nx,ny) in bounds AND not visited AND grid[nx][ny] == 1:
#                 visited.add((nx,ny))
#                 q.append((nx,ny))
#                 size += 1

#     return size

from collections import deque

def largest_island(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    visited = set()
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    max_size = 0

    def bfs(r, c):
        q = deque([(r,c)])
        visited.add((r,c))
        size = 1
        while q:
            x, y = q.popleft()
            for dx, dy in dirs:
                nx, ny = x+dx, y+dy
                if 0 <= nx < rows and 0 <= ny < cols and (nx,ny) not in visited and grid[nx][ny] == 1:
                    visited.add((nx,ny))
                    q.append((nx,ny))
                    size += 1
        return size

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and (r,c) not in visited:
                max_size = max(max_size, bfs(r,c))

    return max_size

print(largest_island(grid))


# PLANE PRACTICE

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

intervals = [[1,2], [3,6], [4, 8], [8, 12]]
def merge(intervals):
    if not intervals:
        return []

    intervals.sort()
    res = [intervals[0]]

    for s,e in intervals[1:]:
        if s <= res[-1][1]:
            res[-1][1] = max(res[-1][1], e)
        else:
            res.append([s,e])
    return res

print(merge(intervals))



def prefix(nums, k):
    curr = 0
    count = {0: 1}
    ans = 0

    for n in nums:
        curr += n

        need = curr - k
        ans += count.get(need, 0)

        count[curr] = count.get(curr, 0) + 1

    return ans


nums = [1,2,1,2,1]
k = 3
print(prefix(nums, k))


heights = [2,1,5,6,2,3]

def largestRectangle(heights):
    stack = [] 
    max_area = 0

    for i, h in enumerate(heights):
        start = i

        # if current bar is shorter → resolve previous
        while stack and stack[-1][1] > h: # Current bar is shorter → previous tall bar can’t extend anymore. Time to compute its final rectangle.
            index, height = stack.pop()
            width = i - index
            max_area = max(max_area, height * width)
            start = index

        stack.append((start, h))

    # clean up remaining
    for index, height in stack:
        width = len(heights) - index
        max_area = max(max_area, height * width)

    return max_area
