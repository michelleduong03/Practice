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
    # TODO: implement your solution
    pass

# Example test
s = "abcabcbb"
print(length_of_longest_substring(s))  # Output: 3


# merge_intervals.py

from typing import List

def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    # TODO: implement your solution
    pass

# Example test
intervals = [[1,3],[2,6],[8,10],[15,18]]
print(merge_intervals(intervals))  # Output: [[1,6],[8,10],[15,18]]
