# DFS Practice Problem 1:
# Count connected components in an undirected graph

def count_components(n, edges):
    # Step 1: build adjacency list
    graph = {i: [] for i in range(n)}
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)

    visited = set()

    # Step 2: DFS
    def dfs(node):
        visited.add(node)
        for nei in graph[node]:
            if nei not in visited:
                dfs(nei)

    # Step 3: count components
    components = 0
    for node in range(n):
        if node not in visited:
            dfs(node)
            components += 1

    return components


# ------------ TEST ------------
n = 5
edges = [(0, 1), (1, 2), (3, 4)]
print("Connected Components:", count_components(n, edges))


# DFS Practice Problem 2:
# Number of Islands in a grid

def num_islands(grid):
    if not grid:
        return 0

    rows = len(grid)
    cols = len(grid[0])
    visited = set()

    def dfs(r, c):
        # stop if out of bounds
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return

        # stop if water or visited
        if grid[r][c] == "0" or (r, c) in visited:
            return

        # mark as visited
        visited.add((r, c))

        # explore 4 directions
        dfs(r+1, c)  # down
        dfs(r-1, c)  # up
        dfs(r, c+1)  # right
        dfs(r, c-1)  # left

    islands = 0

    # scan entire grid
    for r in range(rows):
        for c in range(cols):
            # if we find new land, run DFS
            if grid[r][c] == "1" and (r, c) not in visited:
                dfs(r, c)
                islands += 1

    return islands


# ------------ TEST ------------
grid = [
    ["1","1","0","0"],
    ["1","0","0","1"],
    ["0","0","1","1"]
]
print("Number of Islands:", num_islands(grid))


# DFS Practice Problem 3:
# Check if a path exists in a maze (0=open, 1=wall)

def path_exists(maze, start, end):
    rows = len(maze)
    cols = len(maze[0])

    visited = set()

    def dfs(r, c):
        # 1) Out of bounds
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False

        # 2) Hit wall or visited
        if maze[r][c] == 1 or (r, c) in visited:
            return False

        # 3) If this IS the end → success
        if (r, c) == end:
            return True

        # mark visited
        visited.add((r, c))

        # explore in 4 directions
        if dfs(r+1, c): return True   # down
        if dfs(r-1, c): return True   # up
        if dfs(r, c+1): return True   # right
        if dfs(r, c-1): return True   # left

        # no path found here
        return False

    return dfs(start[0], start[1])


# ------------ TEST ------------
maze = [
    [0,0,1],
    [1,0,1],
    [0,0,0]
]

start = (0,0)
end = (2,2)

print("Path exists:", path_exists(maze, start, end))



def count_closed_islands(grid):
    rows = len(grid)
    cols = len(grid[0])

    visited = set()

    def dfs(r, c):
        # If we leave the grid, NOT closed
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False

        # If it's water or visited, treat as closed up to this point
        if grid[r][c] == 1 or (r, c) in visited:
            return True

        visited.add((r, c))

        # DFS in 4 directions
        up    = dfs(r-1, c)
        down  = dfs(r+1, c)
        left  = dfs(r, c-1)
        right = dfs(r, c+1)

        # A land region is closed only if ALL 4 sides are closed
        return up and down and left and right

    closed_islands = 0

    # Scan grid
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and (r, c) not in visited:
                if dfs(r, c):       # if DFS returns True → it's closed
                    closed_islands += 1

    return closed_islands


# ------------ TEST ------------
grid = [
    [1,1,1,1,1,1],
    [1,0,0,1,0,1],
    [1,0,1,1,0,1],
    [1,1,1,0,0,1],
    [1,0,0,0,1,1],
    [1,1,1,1,1,1],
]

print("Closed islands:", count_closed_islands(grid))



class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def sumNumbers(root):
    def dfs(node, curr):
        # TODO: DFS accumulate value
        pass
    
    return dfs(root, 0)


# Test
root = TreeNode(1,
    TreeNode(2, TreeNode(4), TreeNode(5)),
    TreeNode(3)
)

print(sumNumbers(root))



def letterCombinations(digits):
    if not digits:
        return []

    phone = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz"
    }

    res = []

    def dfs(i, path):
        # TODO: implement DFS to build strings
        pass

    dfs(0, "")
    return res


# Test
print(letterCombinations("23"))
