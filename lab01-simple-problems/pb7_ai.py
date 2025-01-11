# Complexity: O(nlog(n))
def pb7(nums, k):
    nums.sort(reverse=True)
    return nums[k - 1]


# nums = [7, 4, 6, 3, 9, 1]
# k = 2
# result = k_cel_mai_mare_elem(nums, k)
# print(f"Al {k}-lea cel mai mare element este: {result}")

def tests():
    assert(pb7([7, 4, 6, 3, 9, 1], 2) == 7)
    assert (pb7([7, 4, 6, 3, 9, 1], 3) == 6)
    assert (pb7([7, 4, 6, 3, 9, 1], 6) == 1)


tests()
