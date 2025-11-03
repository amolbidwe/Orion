# sets
# Sets are unorderd collection of data items. They store multiple items in a single variable>
# Set items are seprated by commas and enclosed within curly brackets{}. Sets are unchangable.

s = {2, 4, 2, 6}
print(s) 

info = {"Vishwa", 19, False, 5.9, 45}
print(info)
# Items of the set occur in random order and hence they cannot be accessed using index number

# empty set
random = set()
print(type(random))

for value in info:
    print(value)
