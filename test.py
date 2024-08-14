problems_seq_dict = {
    "A": [1,2,3],
    "B": [3,2,1],
    "C": [2,3,1]
}
problems_seq_cache = {problem: [] for problem in problems_seq_dict.keys()}

if problems_seq_cache.values() == [1,2,3]:
    print("True")
else:   
    print("False")



ar = [0,1,2,3,4,5,6,7,8,9, 10, 11, 12]
ws = 4
stride = 2
for i in range(0, len(ar)-ws+1, stride):
    print("seq: ", tuple(ar[i:i+ws]))

    
    data_dict = {
        "key1": (object1, [1, 2, 3]),
        "key2": (object2, [4, 5, 6]),
        "key3": (object3, [7, 8, 9])
    }

    # Adding data to the dictionary
    data_dict["key4"] = (object4, [10, 11, 12])
    data_dict["key5"] = (object5, [13, 14, 15])

    # Adding to the list of a value
    data_dict["key1"][1].append(4)
    data_dict["key2"][1].extend([7, 8, 9])