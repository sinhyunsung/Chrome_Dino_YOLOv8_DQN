# 환경
  
파이썬 3.8.6   
cuda 11.7   
cudnn 8.9.0    
chrome 125.0.6422.112 (크롬 버전에 맞게 chromedriver 넣어줘야함)  


# 게임 실행

python main.py  

# 문제

디텍션 속도 느려서 상태 반영 제대로 안되는듯함  
장애물에 대한 위치 반영이 미흡   
~~DQN 레이어 잘못된점 수정~~    
~~각 행동 reward 수정~~  
게임속도 점점 빨라지는것 반영안됨  

# 결과

## episode 0  
![에피소드0](https://github.com/sinhyunsung/Chrome_Dino_YOLOv8_DQN/assets/37778339/d18033ed-7c4e-4ffd-b4a1-baf85f7e4f64)

## episode 100  
![에피소드100](https://github.com/sinhyunsung/Chrome_Dino_YOLOv8_DQN/assets/37778339/c75a2dcd-9f90-443e-82aa-a923e6670c0d)

## episode 200  
![에피소드200](https://github.com/sinhyunsung/Chrome_Dino_YOLOv8_DQN/assets/37778339/3160ed54-01b7-4b35-89e3-7ed41189157f)

## episode 300  
![에피소드300](https://github.com/sinhyunsung/Chrome_Dino_YOLOv8_DQN/assets/37778339/f4b9ce98-2cfa-47c4-9935-6df102c8fc85)
