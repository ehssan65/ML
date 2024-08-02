import pygame
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import keras
from keras.models import load_model
from tensorflow.keras.optimizers import Adam


@keras.saving.register_keras_serializable()
def mse(y_true,y_pred):
    return keras.losses.mean_squared_error(y_true,y_pred)


# تنظیمات بازی
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)

dis_width = 300
dis_height = 200
snake_block = 10
snake_speed = 100

pygame.init()
dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('Snake Game with Q-learning')

clock = pygame.time.Clock()
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)

# تعریف شبکه عصبی برای Q-learning
# ساخت مدل شبکه عصبی
def create_model():
    optz = Adam()
    model = Sequential([
        Input(shape=(6,)),  # لایه ورودی با 6 ویژگی
        Dense(24, activation='relu'),  # لایه پنهان اول با 24 نرون و تابع فعال‌سازی ReLU
        Dense(24, activation='relu'),  # لایه پنهان دوم با 24 نرون و تابع فعال‌سازی ReLU
        Dense(4, activation='linear')  # لایه خروجی با 4 نرون و تابع فعال‌سازی خطی
    ])
    model.compile(optimizer=optz, loss='mse')
    model.save_weights
    return model

# بررسی وجود فایل مدل ذخیره شده
try:
    model = tf.keras.models.load_model(".//snake_model1.h5",custom_objects={'mse':mse})
    optz = Adam()
    model.compile(optimizer=optz, loss='mse')
    if model:
        print("yeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeees")
        print(model.summary())
        print(model.weights)
except:
    print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    model = create_model()
    

# تعریف بازی اسنیک
class SnakeGameAI:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = dis_width / 2
        self.y = dis_height / 2
        self.x_change = 0
        self.y_change = 0
        self.snake_List = []
        self.Length_of_snake = 1
        self.foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
        self.foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
        self.game_over = False
        self.steps_without_food = 0  # شمارنده برای حرکات بدون خوردن غذا

    def step(self, action):
        self.steps_without_food += 1

        if action == 0:  # left
            self.x_change = -snake_block
            self.y_change = 0
        elif action == 1:  # right
            self.x_change = snake_block
            self.y_change = 0
        elif action == 2:  # up
            self.y_change = -snake_block
            self.x_change = 0
        elif action == 3:  # down
            self.y_change = snake_block
            self.x_change = 0

        self.x += self.x_change
        self.y += self.y_change

        if self.x >= dis_width or self.x < 0 or self.y >= dis_height or self.y < 0:
            self.game_over = True

        snake_Head = [self.x, self.y]
        self.snake_List.append(snake_Head)
        if len(self.snake_List) > self.Length_of_snake:
            del self.snake_List[0]

        for x in self.snake_List[:-1]:
            if x == snake_Head:
                self.game_over = True

        if self.x == self.foodx and self.y == self.foody:
            self.foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
            self.foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
            self.Length_of_snake += 1
            self.steps_without_food = 0  # بازنشانی شمارنده

        # اضافه کردن محدودیت برای حرکات بدون پیشرفت
        if self.steps_without_food > 100:
            self.game_over = True

        return self.get_state(), self.game_over

    def get_state(self):
        state = [self.x, self.y, self.foodx, self.foody, self.x_change, self.y_change]
        return np.array(state)

    def render(self):
        dis.fill(blue)
        pygame.draw.rect(dis, red, [self.foodx, self.foody, snake_block, snake_block])
        for x in self.snake_List:
            pygame.draw.rect(dis, black, [x[0], x[1], snake_block, snake_block])
        pygame.display.update()
#========================================================================================
def get_action():
    for event in pygame.event.get():
        print('\n'+ str(event) +'\n')
        if event.type == pygame.KEYDOWN:
            print('\n key down \n')
            if event.key == pygame.K_LEFT:
                return 0
            elif event.key == pygame.K_RIGHT:
                return 1
            elif event.key == pygame.K_UP:
                return 2
            elif event.key == pygame.K_DOWN:
                return 3

#========================================================================================
# آموزش شبکه عصبی با Q-learning
def train_model_with_q_learning():
    game = SnakeGameAI()
    episodes = 100  # تعداد قسمت‌های آموزشی
    gamma = 0.95  # فاکتور تخفیف
    epsilon = 1.0  # نرخ اکتشاف
    epsilon_min = 0.01  # حداقل نرخ اکتشاف
    epsilon_decay = 0.995  # نرخ کاهش اکتشاف

    for episode in range(episodes):
        game.reset()
        state = game.get_state()
        total_reward = 0
        while not game.game_over:
            #if np.random.rand() <= epsilon:
             #   action = random.randint(0, 3)  # انتخاب حرکت تصادفی
            #else:
             #   action = np.argmax(model.predict(np.array([state]))[0])  # انتخاب حرکت بهینه
#*****************************************************************************
            action = get_action()
#*****************************************************************************            
            next_state, game_over = game.step(action)
            reward = -10 if game_over else 1
            if game.steps_without_food > 70:
                reward -= 10
            if game.x == game.foodx and game.y == game.foody:
                reward += 10
            elif abs(game.x - game.foodx) + abs(game.y - game.foody) < abs(state[0] - state[2]) + abs(state[1] - state[3]):
                reward += 1  # حرکت به سمت غذا
            else:
                reward -= 1  # حرکت دور از غذا

            target = reward + gamma * np.max(model.predict(np.array([next_state]))[0])
            target_f = model.predict(np.array([state]))
            target_f[0][action] = target
            model.fit(np.array([state]), target_f, epochs=1, verbose=0)
            state = next_state
            total_reward += reward

            game.render()
            clock.tick(snake_speed)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode: {episode + 1}, Total reward: {total_reward}, Epsilon: {epsilon}")

# اجرای بازی با شبکه عصبی
def play_game():
    game = SnakeGameAI()
    state = game.get_state()
    while not game.game_over:
        action = np.argmax(model.predict(np.array([state]))[0])
        state, _ = game.step(action)
        game.render()
        clock.tick(snake_speed)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

# آموزش شبکه عصبی با Q-learning
train_model_with_q_learning()
model.save_weights
model.save("snake_model1.h5")

# اجرای بازی با شبکه عصبی
play_game()

pygame.quit()
quit()
