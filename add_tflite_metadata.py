from tflite_support.metadata_writers import writer_utils
from tflite_support.metadata_writers import object_detector

# 路徑配置
modelPath = "lstm_2hand.tflite"
modelWithMetadataPath = "lstm_2hand_metadata.tflite"
labelPath = "labels.txt"

# 模型的資訊
_MODEL_NAME = "lstm_hand_pose"
_MODEL_DESCRIPTION = "This model is predicting hand pose of rubik's cube."
_NORM_MEAN = [0.0] * 84
_NORM_STD = [1.0] * 84

# 創建 Metadata 編寫器
model_buffer = writer_utils.load_file(modelPath)
writer = object_detector.MetadataWriter.create_for_inference(
    model_buffer,
    _NORM_MEAN,
    _NORM_STD,
    [labelPath]
)

# 創建包含 metadata 的模型
with open(modelWithMetadataPath, "wb") as f:
    f.write(writer.populate())

print("Metadata added to the model successfully!")
