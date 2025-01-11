# Complexity: O(n)
def pb6(nums):
    majoritar = nums[0]
    count = 1

    for i in range(1, len(nums)):
        if count == 0:
            majoritar = nums[i]
            count = 1
        elif nums[i] == majoritar:
            count += 1
        else:
            count -= 1

    count = 0
    for num in nums:
        if num == majoritar:
            count += 1
    if count > len(nums) // 2:
        return majoritar
    else:
        return None


# nums = [2, 8, 7, 2, 2, 5, 2, 3, 1, 2, 2]
# rezultat = gaseste_element_majoritar(nums)
# if rezultat is not None:
#     print(f"Elementul majoritar este: {rezultat}")
# else:
#     print("Nu există element majoritar în șirul dat.")

def tests():
    # assert (pb6([2, 8, 7, 5, 3, 1, 2]) == "There is not majority element!")
    try:
        assert (pb6([2, 8, 7, 2, 2, 5, 2, 3, 1, 2, 2]) == 2)
        assert (pb6([2, 8, 3, 7, 3, 3, 1, 3]) == 3)
    except:
        print("An error occurred!")

tests()
