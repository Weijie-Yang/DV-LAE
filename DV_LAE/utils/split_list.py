import numpy as np


def split_list(data_list, split_list):
    sorted_split_list = sorted(split_list)
    result = []
    data_list.pop(0)
    list21 = data_list.copy()
    list22 = data_list.copy()
    for num in sorted_split_list:
        list1=[]
        for item in list21:
            if item <=num:
                list1.append(item)
                list22.remove(item)
        result.append(list1)
        list21=list22
    if len(list21)>0:
        result.append(list21)
    return result

