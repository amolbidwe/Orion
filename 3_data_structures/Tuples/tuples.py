#tuple 
# Tuples are unchangeable
# It is the ordered collection of data items. They store multiple items in a single variable. Tuples are separated by comma and enclosed within round bracket. 
tup = (1, 5, 6, "Violet", 741)
print(type(tup), tup)
print(tup[0])
print(tup[1])
print(tup[2])
print(tup[3])

if 741 in tup:
    print("Yes 741 is present in this tuple")
else:
    print("342 is not present in tuple")
# Here we can do slicing and after slicing new tuple is formed as tuple is unchangeable.

tup2 = tup[1 : 4]
print(tup2)
tup3 = tup[1 : 5 : 2]
print(tup3)
