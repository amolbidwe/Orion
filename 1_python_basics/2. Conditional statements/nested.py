num = 45
if (num < 0):
    print("Number is negative.")

elif (num > 0):
    if(num <= 30):
        print("Number is less than 30")
    elif (num > 30 and num <= 50):
        print("Number is between 30-50")
    else:
        print("Number is greater than 50")
        
else:
    print("Number is zero")
