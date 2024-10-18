def isHandMoving(results):  
    global continuousFeature
    if not hasattr(isHandMoving, "lastFingertips"):
        isHandMoving.lastFingertips = []  
    threshold = [0.15, 0.2]
    fingertipsNodes = [8, 12, 16, 20]  # 4

    leftFingertips = []
    rightFingertips = []
    if results.multi_hand_landmarks:
        for handLandmarks, handed in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            if handed.classification[0].label == "Left":
                leftWrist = [
                    handLandmarks.landmark[0].x,
                    handLandmarks.landmark[0].y,
                ]
                for i in fingertipsNodes:
                    leftFingertips.append(handLandmarks.landmark[i].x)
                    leftFingertips.append(handLandmarks.landmark[i].y)
            elif handed.classification[0].label == "Right":
                for i in fingertipsNodes:
                    rightFingertips.append(handLandmarks.landmark[i].x)
                    rightFingertips.append(handLandmarks.landmark[i].y)
    currentFingertips = leftFingertips + rightFingertips
    for i in range(len(currentFingertips)):
        currentFingertips[i] = currentFingertips[i] - leftWrist[i % 2]
    currentFingertips = organizer.normalizedOneDimensionList(currentFingertips) 

    if len(isHandMoving.lastFingertips)>0:
        for i in range(len(currentFingertips)): 
            fingertipsSAD = abs(currentFingertips[i] - isHandMoving.lastFingertips[i])
            if fingertipsSAD > threshold[i % 2]:
                print(f"fingertipsSAD:{fingertipsSAD},i:{i}")
                isHandMoving.lastFingertips =currentFingertips
                return True
        
 
    isHandMoving.lastFingertips = currentFingertips
    return False

  