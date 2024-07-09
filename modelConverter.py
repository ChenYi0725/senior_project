import tensorflow as tf

# 載入 saved_model 資料夾（假設資料夾名稱為 'saved_model'）
converter = tf.lite.TFLiteConverter.from_saved_model("lstm_2hand_save_model")
tflite_model = converter.convert()

# 保存 TFLite 模型到文件（例如 'model.tflite'）
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
