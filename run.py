from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

# Chrome 드라이버 경로 설정
CHROME_DRIVER_PATH = 'chromedriver'  # Chromedriver 경로로 변경

# Chrome 옵션 설정
options = webdriver.ChromeOptions()
options.add_argument("--disable-extensions")
options.add_argument("--disable-popup-blocking")
options.add_argument("--profile-directory=Default")
options.add_argument("--disable-plugins-discovery")
options.add_argument("--start-maximized")

# 브라우저 실행
driver = webdriver.Chrome(executable_path=CHROME_DRIVER_PATH, options=options)

# Chrome Dino 게임 페이지로 이동
driver.get('chrome://dino')

# 게임 시작을 위해 페이지 클릭
body = driver.find_element_by_tag_name('body')
body.send_keys(Keys.SPACE)

# 게임 제어
try:
    while True:
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
