# python list
# It is ordered collection of data items
# they store multiple itens in single variable
# list items are changable meanings we can alter them after creation
l = [ 3, 5 , 6 ,"Vishwa",True] # list may be consist of multiple data types
#    [0] [1] [2]  [3]    [4]    index  
print(l)
print(l[:])
print(l[1:4])
print(l[1:4:2])  # here 2 is jump index 
print(type(l))
print(l[0])
print(l[1])
print(l[2])

print(l[-3])
print(l[len(l)-3])
print(l[5-3])
print(l[2])


if "Vishwa" in l:
    print("Yes")
else:
    print("No")

if "wa" in "Vishwa":
    print("Yes")
else:
    print("No")
    
lst = [i*i for i in range(10)]
print(lst)
lst = [i*i for i in range(10) if i%2]
print(lst)
lst = [i*i for i in range(4)]
print(lst)
