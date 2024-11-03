import time
startTime = time.time()
counter = 0 
for i in range(100000000):
    counter = counter+1
    endTime = time.time()
    if endTime - startTime > 2:
        break

# if (endTime-startTime)>5:
#     print("long")
print(endTime-startTime)