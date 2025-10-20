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
    seen = {}

    for i, num in enumerate(nums):
        diff = target - num
        if diff in seen:
            return [seen[diff],i]
        seen[num] = i
    
    return None

print("---PROBLEM 13---")
# Example usage
nums = [2, 7, 11, 15]
target = 9
print(two_sum(nums, target))  # Output: [0, 1]

# Problem 14: Group Anagrams
# Group anagrams from a list of strings.
def group_anagrams(strs):
    groups = {}

    for str in strs:
        key = tuple(sorted(str))

        if key not in groups:
            groups[key] = []
        groups[key].append(str)

    return list(groups.values())

print("---PROBLEM 14---")
words = ["eat", "tea", "tan", "ate", "nat", "bat", "tae"]
# Expected Output:
# [
#   ["eat", "tea", "ate"],
#   ["tan", "nat"],
#   ["bat"]
# ]
print(group_anagrams(words))

# Problem 15: Find All Duplicates in an Array
# Return a list of all duplicates in nums.
def find_duplicates(nums):
    seen = {}
    res = []
    for i in nums:
        if i in seen:
            res.append(i)
        else:
            seen[i] = 1
    return res

nums1 = [4, 3, 2, 7, 8, 2, 3, 1] # 2, 3
print("---PROBLEM 15---")
print(find_duplicates(nums1))
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

    for each query, we can loop and check each word to see if each word in names has the prefix
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

# def three_sum(nums): # medium
#     """
#     a + b + c = 0

#     triple for loop but thats not efficient
#     """
#     res = set()
#     nums.sort()
#     for i in range(len(nums)):
#         for j in range(i + 1, len(nums)):
#             for k in range (j + 1, len(nums)):
#                 if nums[i] + nums[j] + nums[k] == 0:
#                     tmp = [nums[i], nums[j], nums[k]]
#                     res.add(tuple(tmp))
#     return [list(i) for i in res]

# OR
def three_sum(nums):
    res = []
    nums.sort()
    
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            
            if total == 0:
                res.append([nums[i], nums[left], nums[right]])
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1
                while left < right and nums[right] == nums[right + 1]:
                    right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    
    return res

# Example usage
nums = [-1, 0, 1, 2, -1, -4]
print("---3SUM---")
print(three_sum(nums))  # Expected: [[-1, -1, 2], [-1, 0, 1]]

def rotate(matrix):
    """
    Rotate the n x n matrix 90 degrees clockwise in-place.
    """
    n = len(matrix)

    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    for i in range(n):
        matrix[i].reverse()
    
    return matrix

# Example usage
matrix1 = [
    [1,2,3],
    [4,5,6],
    [7,8,9]
]

print("---ROTATE IMAGE---")
rotate(matrix1)
print(matrix1)

matrix2 = [
    [5, 1, 9, 11],
    [2, 4, 8, 10],
    [13, 3, 6, 7],
    [15, 14, 12, 16]
]
rotate(matrix2)
print(matrix2)

def gray_code(n):
    res = [0]
    for i in range(n):
        add_val = 2 ** i 
        for x in reversed(res):
            res.append(x + add_val)
    return res

print("---GRAY CODE---")
print(gray_code(1))  # [0, 1]
print(gray_code(2))  # [0, 1, 3, 2]
print(gray_code(3))  # [0, 1, 3, 2, 6, 7, 5, 4]

# EX Problems hackkerrank
from typing import List # for type hints, don't need but good for understanding data types

"""

Question 1: Subarray Sums at Most K

Description:
Given an array of positive integers nums and an integer k, 
return the number of (continuous) subarrays whose sum is at most k.


Example Usage:
Input: nums = [1,2,3], k = 4 → Output: 4
    Explanation: the subarrays with sum ≤ 4 are [1], [2], [1,2], [3].

Input: nums = [2,2,2], k = 2 → Output: 3
    Explanation: [2], [2], [2] each alone.


Notes / hints:
- All numbers are positive.
- You need to count all continuous subarrays (not just distinct sums).
- Think of using a sliding window/two-pointer approach to avoid O(n²) brute force.

"""
def count_subarrays_at_most_k(nums: List[int], k: int) -> int:
    start = 0
    curr_sum = 0
    count = 0

    for end in range(len(nums)):
        curr_sum += nums[end]

        while curr_sum > k:
            curr_sum -= nums[start]
            start += 1

        count += end - start + 1

    return count

# nums = [1,2,3]
# k = 4 
nums = [2,2,2]
k = 2
print("HC-SUBARRAYS")
print(count_subarrays_at_most_k(nums, k))

"""

Question 2: Minimum Flips to Make Binary String Alternating

Description:
Given a binary string s (consisting only of characters '0' and '1'), 
return the minimum number of flips needed so that the string becomes 
alternating, i.e., no two adjacent characters are the same. The target 
string can be either starting with '0' ("010101…") or starting with 
'1' ("101010…"); you should choose whichever requires fewer flips.


Example Usage:
Input: s = "0100" → Output: 1
    Explanation: flip the last 0 to 1, giving "0101".

Input: s = "1110" → Output: 2   
    Explanation: possible alternating strings of same length are "1010" or "0101".
        To reach "1010": flips at positions 0 and 2 → 2 flips.
        To reach "0101": flips at positions 1 and 3 → 2 flips. So result is 2.

        
Notes / hints:
- You can simulate two target patterns and count mismatches.
- Time complexity should be O(n).
- Space complexity should be O(1) extra (not counting input).

"""
def min_flips_alternating(s: str) -> int:
    flips_start_0 = 0
    flips_start_1 = 0
    for i, char in enumerate(s):
        expected0 = '0' if i % 2 == 0 else '1'
        expected1 = '1' if i % 2 == 0 else '0'
        if char != expected0:
            flips_start_0 += 1
        if char != expected1:
            flips_start_1 += 1
    return min(flips_start_0, flips_start_1)

s = "1110" 
print("HC-MINFLIPS") 
print(min_flips_alternating(s))

"""
Question 3: Number of Unique Duplicate Values in an Array

Description:
Given a list of numbers, return the number of unique values that appear more than once in the list.
For example: if the list is [1,2,2,3,3,3,4], the unique values that are duplicates are 2 and 3 → so result = 2.

Example Usage:
Input: nums = [4,3,2,7,8,2,3,1] → Output: 2
    Explanation: The numbers 2 and 3 both appear more than once.

Input: nums = [1,1,2] → Output: 1
    Explanation: Only the number 1 appears more than once.

Notes / hints:
- Use a dictionary or hash map to track counts of each number.
- Then count how many numbers have count > 1.
- Time complexity should be O(n) on average.
- Space complexity is O(n) for the map.

"""
def count_unique_duplicates(nums): 
    seen = {}
    for num in nums:
        if num in seen:
            seen[num] += 1
        else:
            seen[num] = 1

    count = 0
    for val in seen.values():
        if val > 1:
            count += 1
    return count

# nums = [4,3,2,7,8,2,3,1]
# nums = [1,1,2]
nums = [1, 1, 2, 2, 2]
print("HC-UNIQUE")
print(count_unique_duplicates(nums))


"""
Question 4: Prefix Queries on Names List

Description:
Given a list of names (strings) and a list of queries (strings), 
return a list of integers where each integer corresponds to how many names 
the query is a proper prefix of.
"Proper prefix" here means the query appears at the beginning of the name.

Example Usage:
Input:
    names = ["alice", "alex", "albert", "bob"]
    queries = ["al", "bo", "c"]
Output: [3, 1, 0]
    Explanation:
      - "al" is prefix to "alice", "alex", "albert" → count = 3
      - "bo" is prefix to "bob" → count = 1
      - "c" is prefix to none → count = 0

Notes / hints:
- Loop through queries and for each loop through names (nested loop).
- Or optimize using a prefix-map/trie if lists are very large.
- Time complexity for simple nested loops: O(Q * N * L) 
  where Q=num queries, N=num names, L=average name length.
- Space complexity: O(Q + N) (for result + names list).
"""
def prefix_query_count(names, queries):
    res = []
    for query in queries:
        count = 0
        for name in names:
            n = len(query)
            # gives first n letters in the name
            if name[:n] == query:
                count += 1
        res.append(count)
    return res

names = ["alice", "alex", "albert", "bob"]
queries = ["al", "bo", "c"]
print("HC-QUERIES")
print(prefix_query_count(names, queries))

# DP CRASH COURSE
def fib_dp(n): # bottom-up Runtime: O(n), Space: O(n)
    if n <= 1:
        return n
    
    dp = [0] * (n+1)  # dp[i] will store fib(i)
    dp[0] = 0
    dp[1] = 1
    
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

print("FIB-DB")
print(fib_dp(5))  # Output: 5
print(fib_dp(10)) # Output: 55


def climb_stairs(n):
    if n <= 1:
        return 1
    
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

print("CLIMB STAIRS - DP")
print(climb_stairs(3))  # Output: 3
print(climb_stairs(5))  # Output: 8

def climb_stairs_optimized(n):
    if n <= 1:
        return 1
    
    a, b = 1, 1  # dp[0], dp[1]
    
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b

print(climb_stairs_optimized(5))  # Output: 8


"""
Question 1: Maximum Product of Two Numbers in an Array

Description:
Given a list of positive integers nums, return the maximum product you can 
obtain by multiplying any two distinct numbers in the list.

Example Usage:
Input: nums = [3, 4, 5, 2] → Output: 20
    Explanation: 5 * 4 = 20

Input: nums = [1, 10, 2, 6] → Output: 60
    Explanation: 10 * 6 = 60

Notes / hints:
- You can sort the array and multiply the two largest numbers.
- Time complexity should ideally be O(n) if you scan for the two largest numbers.
"""
def max_product(nums):
    max1, max2 = 0, 0
    for num in nums:
        if num > max1:
            max2 = max1
            max1 = num 
        elif num > max2:
            max2 = num
    
    return max1 * max2


nums = [1, 10, 2, 6]
print("MAX PROD")
print(max_product(nums))

"""
Question 2: Count Substrings with Exactly K Distinct Characters

Description:
Given a string s and an integer k, return the number of substrings 
that contain exactly k distinct characters.

Example Usage:
Input: s = "pqpqs", k = 2 → Output: 7
    Explanation: Substrings with exactly 2 distinct chars:
    ["pq","pqp","qp","qpq","pq","qs","pqs"]

Input: s = "aabab", k = 3 → Output: 0
    Explanation: No substring contains exactly 3 distinct characters.

Notes / hints:
- Consider using a sliding window + hashmap to track character counts.
- Time complexity should be O(n) on average.
"""
def count_substrings_k_distinct(s, k):
    # Your code here
    pass
