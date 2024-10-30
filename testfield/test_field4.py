# def isHandMoving(results):  
#     global continuousFeature
#     if not hasattr(isHandMoving, "lastFingertips"):
#         isHandMoving.lastFingertips = []  
#     threshold = [0.15, 0.2]
#     fingertipsNodes = [8, 12, 16, 20]  # 4
#     fingertipsInFeatures = [  # 6,7,
#         14,
#         15,
#         22,
#         23,
#         30,
#         31,
#         38,
#         39,
#         48,
#         49,
#         56,
#         57,
#         64,
#         65,
#         72,
#         73,
#         80,
#         81,
#     ]

#     if len(isHandMoving.lastFingertips) == 0:
#         leftFingertips = []
#         rightFingertips = []
#         if results.multi_hand_landmarks:
#             for handLandmarks, handed in zip(
#                 results.multi_hand_landmarks, results.multi_handedness
#             ):
#                 if handed.classification[0].label == "Left":
#                     leftWrist = [
#                         handLandmarks.landmark[0].x,
#                         handLandmarks.landmark[0].y,
#                     ]
#                     for i in fingertipsNodes:
#                         leftFingertips.append(handLandmarks.landmark[i].x)
#                         leftFingertips.append(handLandmarks.landmark[i].y)
#                 elif handed.classification[0].label == "Right":
#                     for i in fingertipsNodes:
#                         rightFingertips.append(handLandmarks.landmark[i].x)
#                         rightFingertips.append(handLandmarks.landmark[i].y)
#         isHandMoving.lastFeature = leftFingertips + rightFingertips
#         for i in range(len(isHandMoving.lastFeature)):
#             isHandMoving.lastFeature[i] = isHandMoving.lastFeature[i] - leftWrist[i % 2]
#         isHandMoving.lastFeature = organizer.normalizedOneDimensionList(isHandMoving.lastFeature)        
        
#     else:

#     # if len(continuousFeature) > 0:
#     #     lastFingertips = []
#     #     for i in range(len(continuousFeature[-1])):
#     #         lastLeftWrist = [continuousFeature[-1][0], continuousFeature[-1][1]]
#     #         if i in fingertipsInFeatures:
#     #             lastFingertips.append(continuousFeature[-1][i] - lastLeftWrist[i % 2])
        
#         leftFingertips = []
#         rightFingertips = []
#         if results.multi_hand_landmarks:
#             for handLandmarks, handed in zip(
#                 results.multi_hand_landmarks, results.multi_handedness
#             ):
#                 if handed.classification[0].label == "Left":
#                     leftWrist = [
#                         handLandmarks.landmark[0].x,
#                         handLandmarks.landmark[0].y,
#                     ]
#                     for i in fingertipsNodes:
#                         leftFingertips.append(handLandmarks.landmark[i].x)
#                         leftFingertips.append(handLandmarks.landmark[i].y)
#                 elif handed.classification[0].label == "Right":
#                     for i in fingertipsNodes:
#                         rightFingertips.append(handLandmarks.landmark[i].x)
#                         rightFingertips.append(handLandmarks.landmark[i].y)
#         currentFingertips = leftFingertips + rightFingertips
#         for i in range(len(currentFingertips)):
#             currentFingertips[i] = currentFingertips[i] - leftWrist[i % 2]
#         currentFingertips = organizer.normalizedOneDimensionList(currentFingertips)


#         for i in range(len(currentFingertips)): # i = 0???
#             fingertipsSAD = abs(currentFingertips[i] - isHandMoving.lastFingertips[i])
#             if fingertipsSAD > threshold[i % 2]:
#                 # print(f"fingertipsSAD:{fingertipsSAD},i:{i}")
#                 print(f"{currentFingertips[i]},{isHandMoving.lastFingertips[i]},{fingertipsSAD}")
#                 isHandMoving.lastFeature = currentFingertips
#                 return True
#     # else:
#         # return True
#     return False

# #---
# def isHandMoving(results):  
#     global continuousFeature
#     if not hasattr(isHandMoving, "lastFingertips"):
#         isHandMoving.lastFingertips = []  
#     threshold = [0.15, 0.2]
#     fingertipsNodes = [8, 12, 16, 20]  # 4
#     fingertipsInFeatures = [  # 6,7,48,49
#         14,
#         15,
#         22,
#         23,
#         30,
#         31,
#         38,
#         39,
#         56,
#         57,
#         64,
#         65,
#         72,
#         73,
#         80,
#         81,
#     ]

#     if len(continuousFeature) > 0:
#         lastFingertips = []
#         for i in range(len(continuousFeature[-1])):
#             lastLeftWrist = [continuousFeature[-1][0], continuousFeature[-1][1]]
#             if i in fingertipsInFeatures:
#                 lastFingertips.append(continuousFeature[-1][i] - lastLeftWrist[i % 2])
        
#         leftFingertips = []
#         rightFingertips = []
#         if results.multi_hand_landmarks:
#             for handLandmarks, handed in zip(
#                 results.multi_hand_landmarks, results.multi_handedness
#             ):
#                 if handed.classification[0].label == "Left":
#                     leftWrist = [
#                         handLandmarks.landmark[0].x,
#                         handLandmarks.landmark[0].y,
#                     ]
#                     for i in fingertipsNodes:
#                         leftFingertips.append(handLandmarks.landmark[i].x)
#                         leftFingertips.append(handLandmarks.landmark[i].y)
#                 elif handed.classification[0].label == "Right":
#                     for i in fingertipsNodes:
#                         rightFingertips.append(handLandmarks.landmark[i].x)
#                         rightFingertips.append(handLandmarks.landmark[i].y)
#         currentFingertips = leftFingertips + rightFingertips
#         for i in range(len(currentFingertips)):
#             currentFingertips[i] = currentFingertips[i] - leftWrist[i % 2]
#         currentFingertips = organizer.normalizedOneDimensionList(currentFingertips)
#         lastFingertips = organizer.normalizedOneDimensionList(lastFingertips) 

#         for i in range(len(currentFingertips)): 
#             fingertipsSAD = abs(currentFingertips[i] - lastFingertips[i])
#             if fingertipsSAD > threshold[i % 2]:
#                 print(f"fingertipsSAD:{fingertipsSAD},i:{i}")
#                 return True
#     else:
#         print(len(continuousFeature))
#         return True
#     return False
