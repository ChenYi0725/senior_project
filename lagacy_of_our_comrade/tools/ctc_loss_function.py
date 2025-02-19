import tensorflow as tf
from tensorflow.keras.losses import Loss


class CTCLoss(Loss):
    def call(self, yTrue, yPred):
        # y_true: 數值型標籤序列，形狀為 (batch_size, max_text_length)
        # y_pred: 預測的概率分佈，形狀為 (batch_size, time_steps, num_classes)

        # 計算 CTC 損失
        inputLength = tf.ones(shape=(tf.shape(yPred)[0], 1)) * tf.shape(yPred)[1]
        labelLength = tf.ones(shape=(tf.shape(yTrue)[0], 1)) * tf.shape(yTrue)[1]

        ctcLoss = tf.keras.losses.CTCLoss()
        return ctcLoss(yTrue, yPred, input_length=inputLength, label_length=labelLength)
