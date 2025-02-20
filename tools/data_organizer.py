import numpy as np

# from numba import jit


class data_organizer:
    def _cut_first_time_step(self, np_array):
        np_array = np_array[:, 1:, :]
        return np_array

    def _mediapipe_node_to_index(self, input_list):
        index = []
        for i in input_list:
            index.append(i * 2)
            index.append(i * 2 + 1)
        second_hand = []
        for i in index:
            second_hand.append(i + 42)
        index.extend(second_hand)
        return index

    def _remove_palm_node(self, input_list):
        palm = self._mediapipe_node_to_index([0, 1, 5, 9, 13, 17])
        input_list = np.delete(input_list, palm, axis=2)  # 刪除對應的索引(in features)
        return input_list

    def _keep_index_finger_and_thumb(self, inputList):
        targetIndex = np.array(
            self._mediapipe_node_to_index([0, 3, 4, 7, 8 ])
        )
        filtered_array = np.take(inputList, targetIndex, axis=2)
        return filtered_array

    def _keep_index_finger_and_tips(self, inputList):
        finger_and_tips = np.array(
            self._mediapipe_node_to_index([0, 4, 5, 6, 7, 8, 12, 16, 20])
        )
        filtered_array = np.take(inputList, finger_and_tips, axis=2)
        return filtered_array

    def preprocess_exhibit_data(self,input_list):
        input_list = np.array(input_list)
        input_list = self._normalized_with_each_time_steps(input_list)
        input_list = self._get_relative_with_first_time_step(input_list)
        return input_list
    
    # @staticmethod
    # @jit(nopython=True)
    def preprocess_data(self, input_list):
        input_list = np.array(input_list)
        input_list = self._normalized_with_each_time_steps(input_list)
        # inputList = self._getRelativeWithFirstTimeStep(inputList)
        input_list = self._get_relative_location(input_list)
        # inputList = self.getAccelerate(inputList)
        input_list = self._remove_palm_node(input_list)
        return input_list

    def preprocess_for_shirnk_model(self, input_list):
        input_list = np.array(input_list)
        input_list = self._normalized_with_each_time_steps(input_list)
        input_list = self._get_relative_location(input_list)
        input_list = self._keep_index_finger_and_tips(input_list)
        return input_list
    
    def preprocess_for_index_and_Thumb_model(self, input_list):
        input_list = np.array(input_list)
        input_list = self._normalized_with_each_time_steps(input_list)
        input_list = self._get_relative_location(input_list)
        input_list = self._keep_index_finger_and_thumb(input_list)
        return input_list

    @staticmethod
    def _get_relative_location(np_array):  # 以各個時間步的左手腕為基準，輸入:(data number,time step, features)
        for i in range(len(np_array)):
            for j in range(len(np_array[i])):
                origin_x = np_array[i][j][0]
                origin_y = np_array[i][j][1]
                for k in range(len(np_array[i][j])):
                    if k % 2 == 0:
                        np_array[i][j][k] = np_array[i][j][k] - origin_x
                    else:
                        np_array[i][j][k] = np_array[i][j][k] - origin_y
        return np_array

    @staticmethod
    def _normalized_with_each_time_steps(
        input_list,
    ):  # 輸入:(data number,time step, features)

        for i in range(len(input_list)):
            for j in range(input_list.shape[1]):
                input_list[i, j] = (input_list[i, j] - input_list[i, j].min()) / (
                    input_list[i, j].max() - input_list[i, j].min()
                )
        return input_list

    def normalized_one_dimension_list(self, input_list):
        np_input_list = np.array(input_list)
        normalized_list = (np_input_list - np_input_list.min()) / (
            np_input_list.max() - np_input_list.min()
        )
        normalized_list = normalized_list.tolist()
        return normalized_list

    def _get_relative_with_first_time_step(self, np_array):    #以第一個時間步左手腕為基準
        for i in range(len(np_array)):
            origin_x = np_array[i][0][0]
            origin_y = np_array[i][0][1]
            for j in range(len(np_array[i])):
                for k in range(len(np_array[i][j])):
                    if k % 2 == 0:
                        np_array[i][j][k] = np_array[i][j][k] - origin_x
                    else:
                        np_array[i][j][k] = np_array[i][j][k] - origin_y
        return np_array

    def get_data_from_txt(self, file_name):     #從檔案中取得list資料
        with open(f"{file_name}.txt", "r") as file:
            content = file.read()
        result = eval(content)
        return result

    def get_accelerate(self, np_array):   # 取得與上一時間步的差距
        for i in range(len(np_array)):
            for j in reversed(range(len(np_array[i]))):
                for k in reversed(range(len(np_array[i][j]))):
                    if not j < 1:
                        np_array[i][j][k] = np_array[i][j][k] - np_array[i][j - 1][k]
        np_array = self._cut_first_time_step(np_array)
        return np_array

    def find_error_data(self, file_name):  #用於尋找錯誤資料
        target_file = self.get_data_from_txt(file_name)
        error_list = []
        for i in range(len(target_file)):
            if not len(target_file[i]) == 21:
                error_list.append(i)
                continue
            for j in range(len(target_file[i])):
                if not len(target_file[i][j]) == 42:
                    error_list.append(i)
                continue
        return error_list

    def reverse_time_data(self, np_array):
        np_array = [sublist[::-1] for sublist in np_array]
        return np_array
