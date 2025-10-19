# ================================
# 1. Arrays & Strings
# ================================

# Problem 1: Rotate an Array
# Rotate an array of integers 'arr' to the right by 'k' steps.
# Example: arr = [1,2,3,4,5], k=2 -> [4,5,1,2,3]
def rotate_array(arr, k):
    n = len(arr)
    k = k % n  # handle k > n
    return arr[-k:] + arr[:-k]
#       last k elements  rest of arr

arr = [1, 2, 3, 4, 5]
k = 2
print("---PROBLEM 1---")
print(rotate_array(arr, k))

# Problem 2: Longest Substring Without Repeating Characters
# Return the length of the longest substring without repeating characters.
# Example: "abcabcbb" -> 3 ("abc")
def length_of_longest_substring(s):
    start = 0
    max_len = 0
    seen_chars = {}
    for end, char in enumerate(s):
        if char in seen_chars and seen_chars[char] >= start:
            start += 1
        seen_chars[char] = end
        current_len = end - start + 1
        max_len = max(max_len, current_len)

    return max_len

s = "abcabcbb"
print("---PROBLEM 2---")
print(length_of_longest_substring(s))

# Problem 3: Valid Anagram
# Given two strings s and t, return True if t is an anagram of s.
# Runtime is O(n) and space complexity is O(n) --> using two dictionaries
def is_anagram(s, t):
    if len(s) != len(t):
        return False
    
    s_count = {}
    t_count = {}

    for i in s:
        if i not in s_count:
            s_count[i] = 1
        else:
            s_count[i] += 1

    for i in t:
        if i not in t_count:
            t_count[i] = 1
        else:
            t_count[i] += 1
    
    return s_count == t_count

s1 = "listen"
t1 = "silent"
s2 = "aabbcc"
t2 = "abcabc"
s3 = "Listen"
t3 = "Silent"
print("---PROBLEM 3---")
print(is_anagram(s1, t1))
print(is_anagram(s2, t2))
print(is_anagram(s3, t3))

# ================================
# 2. Linked Lists
# ================================

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def create_linked_list(lst):
    if not lst:  # empty list
        return None
    head = ListNode(lst[0])
    curr = head
    for val in lst[1:]:
        curr.next = ListNode(val)
        curr = curr.next
    return head

# Helper: print linked list
def print_linked_list(head):
    curr = head
    res = []
    while curr:
        res.append(curr.val)
        curr = curr.next
    print(res)

# Problem 4: Reverse a Linked List
# Reverse a singly linked list and return the head.
def reverse_linked_list(head):
    curr = head
    prev = None

    while curr:
        next = curr.next
        curr.next = prev
        prev = curr
        curr = next

    return prev

head = create_linked_list([1, 2, 3, 4, 5])

print("---PROBLEM 4---")
print("Original list:")
print_linked_list(head)

reversed_head = reverse_linked_list(head)

print("Reversed list:")
print_linked_list(reversed_head)

# Problem 5: Detect Cycle in a Linked List
# Return True if there is a cycle in the linked list.
def has_cycle(head):
    # visited = set()
    # curr = head

    # while curr:
    #     if curr in visited:
    #         return True
    #     visited.add(curr)
    #     curr = curr.next
    
    # return False

# OR
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
node3 = head.next.next      # node with value 3
node5 = head.next.next.next.next  # node with value 5
node5.next = node3

print("---PROBLEM 5---")
print(has_cycle(head))
head2 = ListNode(1, ListNode(2, ListNode(3)))
print(has_cycle(head2))

# Problem 6: Merge Two Sorted Lists
# Merge two sorted linked lists and return the merged list.
def merge_two_sorted_lists(l1, l2):
    dummy = ListNode(0) # placeholder node
    tail = dummy

    while l1 and l2:
        if l1.val < l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next
    
    if l1:
        tail.next = l1
    if l2:
        tail.next = l2

    return dummy.next

print("---PROBLEM 6---")
l1 = create_linked_list([1, 3, 5])
l2 = create_linked_list([2, 4, 6])

merged1 = merge_two_sorted_lists(l1, l2)
print_linked_list(merged1) 
# ================================
# 3. Trees & Graphs
# ================================

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Problem 7: Binary Tree Level Order Traversal
# Return a list of lists of values for each tree level.
def level_order_traversal(root):
    if not root:
        return []

    queue = [root]
    res = []

    while queue:
        level_nodes = []

        for i in range(len(queue)):
            node = queue.pop(0)
            level_nodes.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            
        res.append(level_nodes)

    return res

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right.right = TreeNode(6)

print("---PROBLEM 7---")
print(level_order_traversal(root))  # Expected: [[1], [2, 3], [4, 5, 6]]

# Problem 8: Shortest Path in Unweighted Graph
# Implement BFS to find the shortest path from source to target in an unweighted graph.
# graph is given as adjacency list, e.g., {0:[1,2], 1:[2], 2:[0,3], 3:[3]}
def shortest_path(graph, start, end):
    if start == end:
        return [start]
    
    from collections import deque
    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        node, path = queue.popleft()

        if node == end:
            return path

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None
    
graph = {
    0: [1, 2],
    1: [2],
    2: [0, 3],
    3: []
}
print("---PROBLEM 8---")
print(shortest_path(graph, 0, 3))  # Output: [0, 2, 3]
print(shortest_path(graph, 1, 3))  # Output: [1, 2, 3]
print(shortest_path(graph, 3, 0))  # Output: None

# Problem 9: Lowest Common Ancestor in BST
# Given a BST root and two nodes p and q, return their lowest common ancestor.
def lowest_common_ancestor(root, p, q):
    pass  # implement your solution


# ================================
# 4. Dynamic Programming
# ================================

# Problem 10: 0/1 Knapsack
# Given weights and values arrays and capacity W, return max value possible.
def knapsack(weights, values, W):
    pass  # implement your solution

weights = [2, 3, 4]
values  = [3, 4, 5]
W = 5

# Problem 11: Longest Common Subsequence
# Return the length of LCS of text1 and text2.
def longest_common_subsequence(text1, text2):
    n = len(text1)
    m = len(text2)

    # Create a table of size (n+1) x (m+1) filled with 0
    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if text1[i - 1] == text2[j - 1]:  # characters match
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:  # take the bigger LCS if we skip a character
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[n][m]

print("---PROBLEM 11---")
print(longest_common_subsequence("abcde", "ace"))  # Output: 3

# Problem 12: Coin Change
# Given coins array and amount, return minimum number of coins to make amount.
def coin_change(coins, amount):
    # dp[a] = minimum coins needed to make amount a
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # 0 coins to make amount 0

    for a in range(1, amount + 1):
        for coin in coins:
            if a - coin >= 0:
                dp[a] = min(dp[a], dp[a - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

print("---PROBLEM 12---")
print(coin_change([1, 2, 5], 11))  # Output: 3 (5 + 5 + 1)
print(coin_change([2], 3))         # Output: -1 (impossible)

# ================================
# 5. Hashing
# ================================

# Problem 13: Two Sum
# Return indices of two numbers that add up to target.
def two_sum(nums, target):
    pass  # implement your solution

# Problem 14: Group Anagrams
# Group anagrams from a list of strings.
def group_anagrams(strs):
    pass  # implement your solution

# Problem 15: Find All Duplicates in an Array
# Return a list of all duplicates in nums.
def find_duplicates(nums):
    pass  # implement your solution


# ================================
# 6. Mathematical Problems
# ================================

# Problem 16: Greatest Common Divisor (GCD)
def gcd(a, b):
    pass  # implement your solution

# Problem 17: Prime Number Check
def is_prime(n):
    pass  # implement your solution

# Problem 18: Count Primes
# Return number of prime numbers less than n.
def count_primes(n):
    pass  # implement your solution


# IBM BANK
def solution(s):
    minflips = 0
    for i in range(0, len(s), 2):
        if s[i]==s[i+1]:
            continue
        else:
            minflips+=1 

    return minflips

print("---MIN FLIP---")
print(solution("10011001"))
print(solution("101011"))

def count_unique_duplicates(nums):
    """
    Return the number of unique duplicate numbers in the list.
    """
    seen = []
    count = 0
    for i in nums:
        if i in seen:
            count += 1
        seen.append(i)
    
    return count

# Example usage:
nums = [1, 2, 3, 2, 3, 4, 5, 5]
print("---UNIQUE DUPES---")
print(count_unique_duplicates(nums))  # Expected output: 3

def prefix_query_count(names, queries):
    """
    For each query, return the number of names it is a proper prefix of.

    for each querie, we can loop and check each word to see if each word in names has the prefix
    inc the count if found
    essentially a double for loop
    """
    res = []
    for query in queries:
        count = 0
        for name in names:
            n = len(query)
            if name[:n] == query:
                count += 1
        res.append(count)
    
    return res

# Example usage:
names = ["alice", "alex", "albert", "bob"]
queries = ["al", "bo", "c"]
print("---PREFIX QUERY---")
print(prefix_query_count(names, queries))  # Expected output: [3, 1, 0]

