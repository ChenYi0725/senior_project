import pose_predict_system
import cv2

def process_video(video_path, output_path=None):
    # 開啟影片檔案
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("無法開啟影片檔案")
        return

    # 如果需要輸出影片，設定寫入器
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 設定影片格式
        fps = cap.get(cv2.CAP_PROP_FPS)  # 取得原始影片的 FPS
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    frame_count = 0
    while True:
        ret, frame = cap.read()

        # 檢查是否成功讀取幀
        if not ret:
            print("影片讀取完成")
            break

        frame_count += 1

        # 將幀輸入 imageHandPosePredict 函式
        try:
            resultString, probabilities, results = pose_predict_system.imageHandPosePredict(frame)  # 假設此函式已定義並可用
            print(f"Result {resultString}: probabilities - {probabilities}")
        except Exception as e:
            print(f"Frame {frame_count}: 發生錯誤 - {e}")

        # 如果需要保存輸出結果，可以在這裡對 frame 或 result 進行標記
        if writer:
            # 可選：在 frame 上疊加結果（根據需求自訂）
            cv2.putText(frame, str(result), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            writer.write(frame)

    # 釋放資源
    cap.release()
    if writer:
        writer.release()
    print("處理完成")

# 假設有影片檔案 video.mp4 和輸出檔案 output.mp4
process_video("video.mp4", "output.mp4")
