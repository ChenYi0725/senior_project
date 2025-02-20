import tensorflow as tf
from tensorflow.keras.losses import Loss


class ctc_loss(Loss):
    def call(self, y_true, y_pred):
        # y_true: 數值型標籤序列，形狀為 (batch_size, max_text_length)
        # y_pred: 預測的概率分佈，形狀為 (batch_size, time_steps, num_classes)

        # 計算 CTC 損失
        input_length = tf.ones(shape=(tf.shape(y_pred)[0], 1)) * tf.shape(y_pred)[1]
        label_length = tf.ones(shape=(tf.shape(y_true)[0], 1)) * tf.shape(y_true)[1]

        ctc_loss = tf.keras.losses.CTCLoss()
        return ctc_loss(y_true, y_pred, input_length=input_length, label_length=label_length)
