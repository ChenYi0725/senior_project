# designated test field for linear interpolation
timeSteps = 21
# ture_data = [(0.505378782749176,0.5589663982391357),(0.507887601852417,0.5595974326133728),(0.5065816044807434,0.5591190457344055),(0.5057867765426636,0.5599932074546814),(0.5066772699356079,0.5598899126052856),(0.5072324872016907,0.5596300363540649),(0.5055049657821655,0.5613670945167542),(0.5047985315322876,0.5674535036087036),(0.5037883520126343,0.5745167136192322),(0.4886852204799652,0.5663190484046936),(0.466178297996521,0.635282039642334),(0.47682881355285645,0.6648396253585815),(0.4801097512245178,0.6749758720397949),(0.48282819986343384,0.6721785664558411),(0.49821364879608154,0.6841291189193726),(0.4970729351043701,0.6974018216133118),(0.49860307574272156,0.6952289938926697),(0.49489736557006836,0.6843584179878235),(0.49938535690307617,0.5988855361938477),(0.5080848336219788,0.5819278359413147),(0.5121835470199585,0.5623421669006348)]
test_data = [0, 2, 4, 6, 8, 20, 21]
# test_data = [0, 2, 4, 6, 8, 10, 11, 12, 14, 16, 18, 20, 21]
test_data = [0, 10, 21]
# for awake Eason:
# fix multi none, when test_data length is less than 12
# find an efficient way to replace 'None' data 

def interpolate_number(returnList): 
    for i in range(len(returnList)):
        if returnList[i] == None:
            counter = 1
            while (returnList[i+counter]) == None:
                counter = counter + 1
            rightValue = returnList[i+counter]
            leftValue = returnList[i-1]
            length = counter +1
            while counter > 0:
                returnList[i+counter-1] = ((rightValue-leftValue)*counter/length)+leftValue
                counter = counter -1 
    return returnList

def linear_interpolation(targetList):  # O(2n)
    global timeSteps
    return_list = [None] * timeSteps
    length = len(targetList)
    return_list[0] = targetList[0]  # head and end
    return_list[20] = targetList[-1]
    for i in range(1, len(targetList) - 1):  # spread the rest of them
        insert_index = ((i * 19) // (length-1) )+ 1
        return_list[insert_index] = targetList[i]
    return_list = interpolate_number(return_list)
    return return_list


print(linear_interpolation(test_data))
