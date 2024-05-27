import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import cv2
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from ultralytics import YOLO
import os
from selenium.webdriver.common.action_chains import ActionChains

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, 256)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)  # (1, 256)
            target = reward
            if not done:
                target = (reward + self.gamma * torch.max(self.model(next_state)).item())
            target_f = self.model(state).detach().clone()
            target_f[0][action] = target
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(state)
            loss = self.criterion(outputs, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

# 환경 초기화
def initialize_environment():
        
    # YOLO 모델 로드
    model = YOLO('runs/detect/train3/weights/best.pt')
    # GPU가 사용 가능한지 확인하고 모델을 GPU로 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if torch.cuda.is_available():
        print("CUDA is available. Running on GPU.")
    else:
        print("CUDA is not available. Running on CPU.")

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
    return driver, body, model

# 게임 오버 감지
def is_game_over(driver, game_over_template):
    screenshot = driver.get_screenshot_as_png()
    screenshot = np.frombuffer(screenshot, np.uint8)
    screenshot = cv2.imdecode(screenshot, cv2.IMREAD_COLOR)
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(screenshot_gray, game_over_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where(result >= threshold)
    return len(loc[0]) > 0

def get_state(driver, model):
    # 화면 캡처
    screenshot = driver.get_screenshot_as_png()
    screenshot = np.frombuffer(screenshot, np.uint8)
    screenshot = cv2.imdecode(screenshot, cv2.IMREAD_COLOR)

    # YOLO 모델로 디텍팅
    results = model(screenshot, verbose=False)[0]

    # 32x32 배열 초기화
    grid_size = 32
    cell_size = 640 // grid_size
    grid = np.zeros((grid_size, grid_size), dtype=int)

    # 검출된 객체 그리기 및 중앙점 계산
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls)
        confidence = box.conf[0]
        label = f"{results.names[cls_id]} {confidence:.2f}"

        # 박스 그리기 및 텍스트 표시
        cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(screenshot, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 중앙점 계산 및 그리드 셀에 표시
        center_x = x1
        center_y = (y1 + y2) // 2
        grid_x = center_x // cell_size
        grid_y = center_y // cell_size

        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
            grid[grid_y, grid_x] = 1

    # 그리드 셀 표시 및 경계선 그리기
    overlay = screenshot.copy()
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
    cv2.waitKey(1)  # 이 줄을 추가하여 OpenCV 창이 표시되도록 함
    time.sleep(0.01)
    return grid.flatten()  # 상태를 1차원으로 펼침

# 메인 함수
if __name__ == "__main__":
    state_size = 32 * 32
    action_size = 3 
    agent = DQNAgent(state_size, action_size)
    model_path = "dqn-dino.pth"
    
    if os.path.exists(model_path):
        print("기존모델 로드")
        agent.load(model_path)
        
    driver, body, yolo_model = initialize_environment()
    game_over_template = cv2.imread('game_over.JPG', 0)  # Grayscale로 로드

    batch_size = 32
    EPISODES = 1000

    actions = ActionChains(driver)


    for e in range(EPISODES):
        state = get_state(driver, yolo_model)
        
        for t in range(5000):
            action = agent.act(state)
            if action == 0:
                actions.key_up(Keys.DOWN).perform()
                body.send_keys(Keys.SPACE)  # Jump
            elif action == 1:
                actions.key_down(Keys.DOWN).perform()  # Duck
            else:
                actions.key_up(Keys.DOWN).perform()

            next_state = get_state(driver, yolo_model) 
            
            if action==0:
                reward = 1
            elif action==1:
                reward = 3
            else:
                reward = 5

            done = is_game_over(driver, game_over_template)
            if done:
                reward = -20  # 게임 오버 시 보상을 0으로 설정
      
            print(action,"->",reward)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, t, agent.epsilon))
                time.sleep(2)
                body.send_keys(Keys.SPACE)  # 게임 재시작
                time.sleep(1.5)
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # Save the model weights after every episode
        if e % 10 == 0:
            agent.save("dqn-dino.pth")

    driver.quit()
    cv2.destroyAllWindows()
