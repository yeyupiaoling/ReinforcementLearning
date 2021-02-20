import cv2
import time
import base64
from gym.spaces import Box
import gym
import numpy as np
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options


class DinoGame:
    def __init__(self, reshape=(1, 150, 450)):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        self.driver = webdriver.Chrome(
            executable_path='C:/Program Files (x86)/Google/Chrome/Application/chromedriver.exe',
            chrome_options=chrome_options
        )
        self.reshape = reshape
        self.driver.maximize_window()
        self.driver.get('http://localhost:5000')  # chrome://dino
        self.driver.execute_script("Runner.config.ACCELERATION=0")
        self.driver.execute_script("document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'")
        self.observation_space = Box(low=0, high=255, shape=reshape)
        self.action_space = gym.spaces.Discrete(2)
        self.last_score = 0
        self.reset()
        self.jump()

    def step(self, action):
        # operate T-Rex according to the action
        if action == 0:
            pass
        elif action == 1:
            self.jump()
        elif action == 2:
            self.bowhead()
        # get score
        score = self.state('score')
        reward = score - self.last_score
        self.last_score = score
        # whether die or not
        if self.state('crashed'):
            self.reset()
            is_dead = True
            reward = -10
        else:
            is_dead = False
        # get game image
        image = self.screenshot()
        image = self.preprocess(image)
        return image, reward, is_dead, {}

    def state(self, type_):
        assert type_ in ['crashed', 'playing', 'score']
        if type_ == 'crashed':
            return self.driver.execute_script("return Runner.instance_.crashed;")
        elif type_ == 'playing':
            return self.driver.execute_script("return Runner.instance_.playing;")
        else:
            digits = self.driver.execute_script("return Runner.instance_.distanceMeter.digits;")
            score = ''.join(digits)
            return int(score)

    def jump(self):
        self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def bowhead(self):
        self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)

    def reset(self):
        self.last_score = 0
        self.driver.execute_script("Runner.instance_.restart();")
        time.sleep(0.2)
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def close(self):
        self.driver.close()

    def screenshot(self, area=(0, 0, 150, 450)):
        image_b64 = self.driver.execute_script(
            "canvasRunner = document.getElementById('runner-canvas'); return canvasRunner.toDataURL().substring(22)")
        image = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[area[0]: area[2], area[1]: area[3]]
        return image

    def preprocess(self, image):
        image = cv2.resize(image, (self.reshape[2], self.reshape[1]))
        _, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
        image = np.expand_dims(image, axis=0)
        return image
