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
    visited = set()
    slow = head
    fast = head

    curr = head
    while curr:
        if curr in visited:
            return True
        visited.add(curr)
        curr = curr.next
    
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
    pass  # implement your solution

print("---PROBLEM 6---")

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
