import time
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO('runs/detect/train3/weights/best.pt')

# Chrome 옵션 설정
options = webdriver.ChromeOptions()
options.add_argument("--disable-extensions")
options.add_argument("--disable-popup-blocking")
options.add_argument("--profile-directory=Default")
options.add_argument("--disable-plugins-discovery")
options.add_argument("--start-maximized")

driver = webdriver.Chrome(executable_path="chromedriver", options=options)

# 브라우저 크기 조정 (640x640)
driver.set_window_size(640, 640)

# Chrome Dino 게임 페이지로 이동
try:
    driver.get('chrome://dino')
except:
    pass

# 게임 시작을 위해 페이지 클릭
body = driver.find_element_by_tag_name('body')
body.send_keys(Keys.SPACE)

# 게임 제어 및 화면 디텍팅
try:
    while True:
        # 화면 캡처
        screenshot = driver.get_screenshot_as_png()
        screenshot = np.frombuffer(screenshot, np.uint8)
        screenshot = cv2.imdecode(screenshot, cv2.IMREAD_COLOR)

        # YOLO 모델로 디텍팅
        results = model(screenshot)[0]

        # 16x16 배열 초기화
        grid_size = 16
        cell_size = 640 // grid_size
        grid = np.zeros((grid_size, grid_size), dtype=int)

        overlay = screenshot.copy()

        # 검출된 객체 그리기 및 중앙점 계산
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            confidence = box.conf[0]
            label = f"{results.names[cls_id]} {confidence:.2f}"

            # 박스 그리기
            cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(screenshot, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 중앙점 계산 및 그리드 셀에 표시
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            grid_x = center_x // cell_size
            grid_y = center_y // cell_size

            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                grid[grid_y, grid_x] = 1

        # 그리드 셀 표시 및 경계선 그리기
        for i in range(grid_size):
            for j in range(grid_size):
                start_x = j * cell_size
                start_y = i * cell_size
                end_x = start_x + cell_size
                end_y = start_y + cell_size
                
                if grid[i, j] == 1:
                    cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 255), -1)

                # 경계선 그리기
                cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), 1)

        # 반투명 오버레이 적용
        alpha = 0.4  # 반투명도 설정
        cv2.addWeighted(overlay, alpha, screenshot, 1 - alpha, 0, screenshot)

        # 결과 이미지 표시
        cv2.imshow("Dino Game Detection", screenshot)

        # ESC 키를 누르면 루프 중단
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
        # Dino 점프
        body.send_keys(Keys.SPACE)
        time.sleep(2)

        # Dino 수그리기
        body.send_keys(Keys.DOWN)
        time.sleep(1)
        
except KeyboardInterrupt:
    pass

# 브라우저 종료
driver.quit()
cv2.destroyAllWindows()
