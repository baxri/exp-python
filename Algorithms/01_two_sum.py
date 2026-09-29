"""
Two Sum

Given a list of integers `nums` and an integer `target`, return the indices
of the two numbers such that they add up to `target`.

You may assume that each input has exactly one solution, and you may not use
the same element twice. Return the indices in any order.

Example 1:
    Input: nums = [2, 7, 11, 15], target = 9
    Output: [0, 1]
    Explanation: nums[0] + nums[1] == 9

Example 2:
    Input: nums = [3, 2, 4], target = 6
    Output: [1, 2]

Constraints:
    - 2 <= len(nums) <= 10_000
    - -1_000_000_000 <= nums[i] <= 1_000_000_000
    - Exactly one valid answer exists.
"""

def two_sum(nums: list[int], target: int) -> list[int]:
    for n in nums:
        remaining = target - n
        if remaining in nums:
            return [nums.index(n), nums.index(remaining)]

def two_sum_range(nums: list[int], target: int) -> list[int]:
    for i in range(len(nums)):
        remaining = target - nums[i]
        if remaining in nums[i + 1:]:
            return [i, nums.index(remaining, i + 1)]

if __name__ == "__main__":
    print(two_sum([2, 7, 11, 15], 9))  # expected [0, 1]
    print(two_sum([3, 2, 4], 6))       # expected [1, 2]



print([2, 7, 11, 15][0:4])