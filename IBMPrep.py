# ================================
# 1. Arrays & Strings
# ================================

# Problem 1: Rotate an Array
# Rotate an array of integers 'arr' to the right by 'k' steps.
# Example: arr = [1,2,3,4,5], k=2 -> [4,5,1,2,3]
def rotate_array(arr, k):
    for i in range(len(arr)):
        arr[i] = i + k
    return arr

arr = [1, 2, 3, 4, 5]
k = 2
print("---PROBLEM 1---")
print(rotate_array(arr, k))

# Problem 2: Longest Substring Without Repeating Characters
# Return the length of the longest substring without repeating characters.
# Example: "abcabcbb" -> 3 ("abc")
def length_of_longest_substring(s):
    pass  # implement your solution

# Problem 3: Valid Anagram
# Given two strings s and t, return True if t is an anagram of s.
def is_anagram(s, t):
    pass  # implement your solution


# ================================
# 2. Linked Lists
# ================================

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Problem 4: Reverse a Linked List
# Reverse a singly linked list and return the head.
def reverse_linked_list(head):
    pass  # implement your solution

# Problem 5: Detect Cycle in a Linked List
# Return True if there is a cycle in the linked list.
def has_cycle(head):
    pass  # implement your solution

# Problem 6: Merge Two Sorted Lists
# Merge two sorted linked lists and return the merged list.
def merge_two_sorted_lists(l1, l2):
    pass  # implement your solution


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
    pass  # implement your solution

# Problem 8: Shortest Path in Unweighted Graph
# Implement BFS to find the shortest path from source to target in an unweighted graph.
# graph is given as adjacency list, e.g., {0:[1,2], 1:[2], 2:[0,3], 3:[3]}
def shortest_path(graph, start, end):
    pass  # implement your solution

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

# Problem 11: Longest Common Subsequence
# Return the length of LCS of text1 and text2.
def longest_common_subsequence(text1, text2):
    pass  # implement your solution

# Problem 12: Coin Change
# Given coins array and amount, return minimum number of coins to make amount.
def coin_change(coins, amount):
    pass  # implement your solution


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
