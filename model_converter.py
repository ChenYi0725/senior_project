import keras

# 載入 .keras 格式的模型
model1 = keras.models.load_model("lstm_index_thumb_model.keras")
model2 = keras.models.load_model("the_precious_working_model/lstm_2hand_noCTC_60Features.keras")

# 儲存為 .h5 格式
model1.save("lstm_index_thumb_model.h5")
model2.save("the_precious_working_model/lstm_2hand_noCTC_60Features.h5")

print("模型已成功轉換為 .h5 格式")
