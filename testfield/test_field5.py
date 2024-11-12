def _MediapipeNodeToIndex(inputList):
    index= []
    for i in inputList: 
        index.append(i*2)
        index.append(i*2+1)
    secondHand = []
    for i in index:
        secondHand.append(i+42)
    index.extend(secondHand)
    return index

print(_MediapipeNodeToIndex([0,1,5,9,13,17]))