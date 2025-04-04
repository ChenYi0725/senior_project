## 1.重要檔案介紹
- **data_set_2hands**
    - 所有原始資料的存放處，其中的reference_table.txt內紀錄了檔案名稱
    與轉動符號的對應關係
    - 對應關係如下
    
    | 名稱                     | 符號 |
    |--------------------------|------|
    | left_down                | L'   |
    | left_up                  | L    |
    | right_down               | R    |
    | right_up                 | R'   |
    | top_left                 | U    |
    | top_right                | U'   |
    | bottom_left              | D'   |
    | bottom_right             | D    |
    | front_clockwise          | F    |
    | front_counter_clockwise  | F'   |
    | back_clockwise           | B'   |
    | back_counter_clockwise   | B    |
    
- **lstm_2hand_save_model**
    - 儲存為 TensorFlow SavedModel 格式的 LSTM 模型



- **tools**
    - 包含本專案可能會重複使用到的各種工具
        - camera
            - 負責影像取得的工作
        - data_examiner
            - 檢查各資料集的資料數量，以及是否有缺失
        - data_organizer
            - 將資料進行處理以便進行其他工作，如模型訓練、動作預測
        - plot_drawer
            - 將資料以視覺化方式表現，檢查是否為有效資料
        - recorder
            - 負責控制手部節點的紀錄時機以及長度

- **hand_tracker**
    - 用於收集資料
    - 按下左鍵可開始記錄，按右鍵刪除最近一筆紀錄
    - 左上角顯示當前FPS，右上角顯示已記錄的資料數
    - 若雙手都位於畫面上則中間會顯示Exist 表示可接收雙手的訊息

- **lstm_2hand_model.h5**
    - 儲存為h5 格式的 LSTM 模型
- **lstm_2hand_model.keras**
    - 儲存為keras 格式的 LSTM 模型
- **lstm_model_trainer**
    - 用於訓練LSTM 模型，執行後會自動從dataset裡面找資料，並將模型儲存於目錄
- **model_converter**
    - 用於將模型轉化成tflite 格式
- **pose_predict_2hands**
    - 執行後可以直接以電腦鏡頭預測魔術方塊的轉動
- **result.txt**
    - 用於存放來自hand_tracker 所收集的資料
- **pose_predict_system、where_the_magic_happend**
    - 這兩者功能相同，用於實際使用時的前處理與後處理。
- **pose_predict**
    - 用於測試模型的檔案，透過tools.camera 取得影像後，由pose_predict_system處理。
- **其他檔案**
    - 開頭為stop_moving，此為用於訓練停止與移動的模型，後因效果不彰所以廢棄。
    - 開頭為thumb_index，此為只保留食指與大拇指的模型，訓練速度快，但效果不佳。
    - 包含9move、small_model、shrink 的檔案為，訓練過程只包含9個動作(去除B、B'、D、D')
    - 包含waving、exhibit 的檔案，皆為專題展當天用於展示用的模型，只包含水平和垂直的動作，透過左右手與握拳與否分出共8個動作。
    - video_predicter 為透過chat gpt 修改pose_predict 後的檔案，可透過輸入影片取得判斷結果，但尚未驗證檔案可行性。
    - model_evaluator 在模型訓練完後會生成混淆矩陣、損失值的收斂情形。
    - 所有名稱帶有test的檔案、以及test_field 中所有檔案，皆為專案進行過程中，用於測試python 語法或是測試function 輸出結果的檔案，可以直接無視。
---
## 2.如何蒐集資料 
1. 執行hand_tracker.py 
2. 若雙手同時存在於螢幕上時，可左鍵點擊螢幕可開始錄影，此時右上角會顯示紅圈，即代表此時雙手的節點會被記錄，若此時雙手拿開會暫停，直到畫面上重新出現兩隻手，此時建議刪除本次錄製資料
3. 若要刪除上次錄製的資料，可對畫面點擊右鍵，此時右上角的times 會減1
4. 當要結束hand_tracker 時，直接輸入q 即可
5. 程式結束後會將本次所錄製的所有資料存入result.txt
6. result 內應會包含一個少了兩個中括弧(最前與最後各一個)的三維矩陣，最外層為本次紀錄的各筆資料，中間為每筆資料的時間步，最內層為每時間步內的特徵
7. 複製result 內所有文字，並將其貼到需要新增的資料庫內，切記保持三維矩陣結構完整，建議貼在dataset最後面  ( [[[...]]*] 星號的位置)，且在新資料前加上一個逗號
8. 當新資料貼上後可以執行data_examiner.py, 就可以確認新資料格式是否有誤，以及出現錯誤的資料有哪些 



***如果上述檔案未出現在該專案中，代表該檔案不再被使用，已經被刪除；如果有未出現在上方的檔案出現在該專案中，代表該檔案主要用於輔助模型訓練、調整，可直接無視。** 