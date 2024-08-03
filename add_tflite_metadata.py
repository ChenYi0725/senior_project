from tflite_support import metadata
from tflite_support.metadata_writers import writer_utils
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import MetadataWriter

# 路徑配置
modelPath = "lstm_2hand.tflite"
modelWithMetadataPath = "lstm_2hand_metadata.tflite"
labelPath = "labels.txt"
# ----------------------
# 模型的資訊
_MODEL_NAME = "lstm_hand_pose"
_MODEL_DESCRIPTION = "This model is predicting hand pose of rubik's cube."
writer = object_detector.MetadataWriter.create_for_inference(
    writer_utils.load_file(modelPath), [labelPath], _MODEL_NAME, _MODEL_DESCRIPTION
)

# 創建包含 metadata 的模型
with open(modelWithMetadataPath, "wb") as f:
    f.write(writer.populate())

print("Metadata added to the model successfully!")
