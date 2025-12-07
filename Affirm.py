# two_sum.py

from typing import List

def two_sum(nums: List[int], target: int) -> List[int]:
    # for i in range(len((nums))):
    #     for j in range (i + 1, len(nums)):
    #         if nums[i] + nums[j] == target:
    #             return [i, j]
    # return []
    seen = {}

    for i, val in enumerate(nums):
        difference = target - val
        if difference in seen:
            return [seen[difference], i]
        seen[val] = i
    return []

# Example test
# nums = [2, 7, 11, 15]
nums = [1, 5, 8, 9]
target = 9
print(two_sum(nums, target))  # Output: [0, 1]


# longest_substring.py

def length_of_longest_substring(s: str) -> int:
    seen = set() # O(n)
    start = 0
    max_len = 0

    for end in range(len(s)):
        while s[end] in seen:
            seen.remove(s[start])
            start += 1
        seen.add(s[end])
        max_len = max(max_len, end - start + 1)
    
    return max_len

# Example test
s = "abcabcbb"
print(length_of_longest_substring(s))  # Output: 3


# merge_intervals.py

from typing import List

def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    if not intervals:
        return []
    
    intervals.sort()
    
    merged = [intervals[0]]
    
    for curr in intervals[1:]:
        last = merged[-1]
        
        # Check overlap
        if curr[0] <= last[1]:
            # Merge
            last[1] = max(last[1], curr[1])
        else:
            merged.append(curr)
    
    return merged

# Example test
intervals = [[1,3],[2,6],[8,10],[15,18]]
print(merge_intervals(intervals))  # Output: [[1,6],[8,10],[15,18]]


from typing import List

def consolidate_transactions(transactions: List[List[int]]) -> List[List[int]]:
    # Sort by customerId, then timestamp
    transactions.sort()   # natural sort works: [customerId, amount, timestamp]
    
    merged = []
    curr_id, curr_amount, curr_time = transactions[0]
    
    for i in range(1, len(transactions)):
        t_id, t_amount, t_time = transactions[i]
        
        # If same customer AND within 60 seconds → merge
        if t_id == curr_id and t_time - curr_time <= 60:
            curr_amount += t_amount   # accumulate amount
            # timestamp stays the earliest one
        else:
            merged.append([curr_id, curr_amount, curr_time])
            curr_id, curr_amount, curr_time = t_id, t_amount, t_time
    
    # append last group
    merged.append([curr_id, curr_amount, curr_time])
    
    return merged

# Test
transactions = [
    [1, 20, 100],
    [1, 15, 120],
    [2, 30, 200],
    [1, 40, 160],
    [2, 10, 250]
]

print(consolidate_transactions(transactions))
# Expected:
# [
#     [1, 35, 100],
#     [1, 40, 160],
#     [2, 40, 200]
# ]
