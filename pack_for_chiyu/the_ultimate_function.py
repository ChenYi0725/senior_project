import pose_predict_system


def picture_in_result_out(RGBImage):
    resultString,probabilities = pose_predict_system.imageHandPosePredict(RGBImage)
    return resultString,probabilities
