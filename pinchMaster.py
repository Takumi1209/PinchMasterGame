import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Mediapipeのセットアップ
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ランダムな位置にオブジェクトを生成する関数
def generate_random_position(frame):
    h, w, _ = frame.shape
    return random.randint(50, w-50), random.randint(50, h-50)

# 距離を計算する関数
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# ゲームの初期設定
game_duration = 30  # 制限時間（秒）
score = 0

# ゲームの状態
GAME_IDLE = 0
GAME_COUNTDOWN = 1
GAME_RUNNING = 2
GAME_OVER = 3

game_state = GAME_IDLE

# カウントダウンの時間
countdown_duration = 3
countdown_start_time = None

# 開始時刻を設定
start_time = None

# アニメーションのための設定
animation_frame_count = 10
animation_current_frame = 0
animation_position = None

# ゲームの初期化
def initialize_game():
    global score, game_state, start_time, animation_current_frame
    score = 0
    game_state = GAME_IDLE
    start_time = None
    animation_current_frame = 0

# マウスクリックイベントのハンドラ関数
def mouse_click(event, x, y, flags, param):
    global game_state, countdown_start_time
    if event == cv2.EVENT_LBUTTONDOWN:
        if game_state == GAME_IDLE:
            # カウントダウン開始
            countdown_start_time = time.time()
            game_state = GAME_COUNTDOWN
        elif game_state == GAME_OVER:
            # リトライ
            initialize_game()

# OpenCVのセットアップ
cap = cv2.VideoCapture(0)
cv2.namedWindow('Pinch Master')
cv2.setMouseCallback('Pinch Master', mouse_click)

# 初期オブジェクトの位置と速度
object_pos = generate_random_position(cap.read()[1])

# オブジェクトの速度を設定
velocity = [random.choice([-15, 15]), random.choice([-15, 15])]  # 初期速度を速く設定

# 速度の調整関数
def set_random_velocity():
    return [random.choice([-20, -17, -15, 15, 17, 20]), random.choice([-20, -17, -15, 15, 17, 20])]  # 速度をさらに速く設定

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 鏡のように反転
        frame = cv2.flip(frame, 1)
        
        # BGR画像をRGBに変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # フレームを処理して手のランドマークを検出
        results = hands.process(rgb_frame)
        
        if game_state == GAME_COUNTDOWN:
            # カウントダウン表示
            countdown_time = int(time.time() - countdown_start_time)
            if countdown_time >= countdown_duration:
                game_state = GAME_RUNNING
                start_time = time.time()
            else:
                # テキストをセンターに合わせ、少し大きくする
                text = f'{countdown_duration - countdown_time}'
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = (frame.shape[0] + text_size[1]) // 2
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        elif game_state == GAME_RUNNING:
            # 手のランドマークを描画
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # 親指の先端と人差し指の先端の座標を取得
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    thumb_tip_pos = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
                    index_tip_pos = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
                    
                    # 親指の先端と人差し指の先端を結ぶ線を描画
                    cv2.line(frame, thumb_tip_pos, index_tip_pos, (0, 165, 255), 2)
                    
                    # オブジェクトと親指の先端および人差し指の先端が重なっているか確認
                    if (calculate_distance(thumb_tip_pos, object_pos) < 45 and 
                        calculate_distance(index_tip_pos, object_pos) < 45):
                        score += 1
                        animation_position = object_pos
                        animation_current_frame = 1
                        object_pos = generate_random_position(frame)
                        velocity = set_random_velocity()  # 新しいランダムな速度を設定
        
            # オブジェクトを移動
            object_pos = (object_pos[0] + velocity[0], object_pos[1] + velocity[1])
            
            # オブジェクトが画面の端に当たった場合、速度を反転
            if object_pos[0] <= 0 or object_pos[0] >= frame.shape[1]:
                velocity[0] = -velocity[0]
            if object_pos[1] <= 0 or object_pos[1] >= frame.shape[0]:
                velocity[1] = -velocity[1]
            
            # 残り時間を計算
            elapsed_time = int(time.time() - start_time)
            remaining_time = game_duration - elapsed_time
            
            # 残り時間が0になったらゲーム終了
            if remaining_time <= 0:
                game_state = GAME_OVER
                continue
            
            # オブジェクトを描画
            cv2.circle(frame, object_pos, 30, (0, 255, 0), -1)
            
            # スコアと残り時間を表示
            cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f'Time: {remaining_time}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if animation_current_frame > 0:
            radius = animation_current_frame * 5
            cv2.circle(frame, animation_position, radius, (0, 255, 255), 2)
            animation_current_frame += 1
            if animation_current_frame > animation_frame_count:
                animation_current_frame = 0
        
        elif game_state == GAME_OVER:
            text = f'Score: {score}'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] // 2)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            retry_text = 'Click to Retry'
            retry_text_size = cv2.getTextSize(retry_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            retry_text_x = (frame.shape[1] - retry_text_size[0]) // 2
            retry_text_y = text_y + 50
            cv2.putText(frame, retry_text, (retry_text_x, retry_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        elif game_state == GAME_IDLE:
            # ゲームタイトルを表示
            title_text = 'Pinch Master'
            title_text_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            title_text_x = (frame.shape[1] - title_text_size[0]) // 2
            title_text_y = (frame.shape[0] // 2) - 50
            cv2.putText(frame, title_text, (title_text_x, title_text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            
            start_text = 'Click to Start'
            start_text_size = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            start_text_x = (frame.shape[1] - start_text_size[0]) // 2
            start_text_y = (frame.shape[0] // 2)
            cv2.putText(frame, start_text, (start_text_x, start_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Pinch Master', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
