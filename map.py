# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import random

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
import os
import torch
import torch.optim as optim
# Importing the Dqn object from our AI in ai.py
from ai import Dqn
from ai import Network

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1203')
Config.set('graphics', 'height', '678')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5,3,0.8)
brain1 = Dqn(5,3,0.8)

# model_path_brain = "last_brain.pth"
# if os.path.exists(model_path_brain):
#     print(f"Loading pretrained model for brain from {model_path_brain}...")
#     brain.load()
# else:
#     print(f"No pretrained model found for brain. Starting fresh.")
action2rotation = [0,5,-5]
action2rotation1 = [0,5,-5]
last_reward = 0
last_reward1 = 0
scores = []
scores1 = []
im = CoreImage("./images/MyMASK12.png")


# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global goal_x1
    global goal_y1
    global first_update
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/Mymask2.png").convert('L')
    sand = np.asarray(img)/255
    goal_x = 252
    goal_y = 75
    goal_x1 = 252
    goal_y1 = 75
    first_update = False
    global swap
    swap = 0
    global swap1
    swap1 = 0


# Initializing the last distance
last_distance = 0
last_distance1 = 0

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        # Getting position of the car
        # Velocity Vector + current position (x & y coordinates).
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        
        # Density of sand around sensor-1 (in a 20x20 square)
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos

        # Density of sand around sensor-2 (in a 20x20 square)
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos

        # Density of sand around sensor-3 (in a 20x20 square)
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos

        # sensor1_x is the x co-ordinate of sensor1
        # sensor1_y is the y co-ordinate of sensor1
        # (sensor1_x + 10 & sensor1_x - 10, sensor1_y + 10 & sensor1_y + 10 gets us boundary we consider for measuring sand density)
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.

        # if the car is around the edges of the map, make sand density sensor value 10 to make this state distinct from a state in just sand and not around edges.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 10.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 10.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 10.
        
class Car1(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        # Getting position of the car
        # Velocity Vector + current position (x & y coordinates).
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        
        # Density of sand around sensor-1 (in a 20x20 square)
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos

        # Density of sand around sensor-2 (in a 20x20 square)
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos

        # Density of sand around sensor-3 (in a 20x20 square)
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos

        # sensor1_x is the x co-ordinate of sensor1
        # sensor1_y is the y co-ordinate of sensor1
        # (sensor1_x + 10 & sensor1_x - 10, sensor1_y + 10 & sensor1_y + 10 gets us boundary we consider for measuring sand density)
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 10.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 10.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 10.

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass
class Ball4(Widget):
    pass
class Ball5(Widget):
    pass
class Ball6(Widget):
    pass

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    car1 = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    ball4 = ObjectProperty(None)
    ball5 = ObjectProperty(None)
    ball6 = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Game, self).__init__(**kwargs)
        self.penalty_counter_car1 = 0  # Đếm số lần xe car1 chạm biên liên tiếp
        self.penalty_counter_car = 0   # Đếm số lần xe car chạm biên liên tiếp
        self.max_penalty = 5         # Giới hạn số lần chạm biên trước khi reset vị trí xe
       

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)
        self.car1.center = self.center
        self.car1.velocity = Vector(6, 0)

    def update(self, dt):
        global brain
        global brain1
        global last_reward
        global last_reward1    
        global scores
        global scores1
        global last_distance
        global last_distance1
        global goal_x
        global goal_y
        global goal_x1
        global goal_y1
        global longueur
        global largeur
        global swap
        global swap1
        

        longueur = self.width
        largeur = self.height

        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y

        xx1 = goal_x1 - self.car1.x
        yy1 = goal_y1 - self.car1.y

        # One of most important states, i.e orientation. 
        # It is the angle between the axis of car and the line joining car to the goal.
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        orientation1 = Vector(*self.car1.velocity).angle((xx1,yy1))/180.

        # State contains, signal 1, signal 2, signal 3, orientation & -orientation
        # Signal 1 detects sand density on left side, signal 2 detects in the center and signal 3 on the right.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        last_signal1 = [self.car1.signal1, self.car1.signal2, self.car1.signal3, orientation1, -orientation1]

        action = brain.update(last_reward, last_signal)
        action1 = brain1.update(last_reward1, last_signal1)
        

        scores.append(brain.score())
        scores1.append(brain1.score())

        rotation = action2rotation[action]
        rotation1 = action2rotation1[action1]

        self.car.move(rotation)
        self.car1.move(rotation1)

        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        
        distance1 = np.sqrt((self.car1.x - goal_x1)**2 + (self.car1.y - goal_y1)**2)

        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        self.ball4.pos = self.car1.sensor1
        self.ball5.pos = self.car1.sensor2
        self.ball6.pos = self.car1.sensor3

        if 0 <= int(self.car.x) < sand.shape[0] and 0 <= int(self.car.y) < sand.shape[1]:
            if sand[int(self.car.x),int(self.car.y)] > 0:
                self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
                last_reward = -10
                self.penalty_counter_car += 1

              
                # Nếu xe car1 chạm biên quá nhiều lần, đưa xe về điểm xuất phát
                if self.penalty_counter_car >= self.max_penalty:
                    self.car.center = self.center   # Đưa car về vị trí trung tâm
                    self.car.velocity = Vector(0, 0)          # Đặt lại vận tốc của xe
                    self.car.angle += random.choice([45, 90, -45])  # Quay xe 45 độ để rời khỏi tường
                    if self.car.angle >= 360:  # Đảm bảo góc luôn trong khoảng [0, 360]
                        self.car.angle -= 360
                
                    self.penalty_counter_car = 0             # Đặt lại bộ đếm cho car1
            else: # otherwise
                self.car.velocity = Vector(2, 0).rotate(self.car.angle)
                last_reward = 3
                self.penalty_counter_car = 0 

                if distance < last_distance:
                    last_reward += 0.2 + (1 / distance + 0.01) * 10
                else:
                    last_reward -= -0.2 * (distance - last_distance)
        else:
            # Nếu xe car1 vượt khỏi giới hạn của mảng sand, đặt lại vị trí và áp dụng hình phạt nặng
            self.car1.center = self.center  # Đưa xe về vị trí trung tâm
            self.car1.velocity = Vector(0, 0)  # Đặt lại vận tốc của xe
            last_reward1 = -10  # Hình phạt lớn hơn khi vượt ra ngoài giới hạn
            

        if distance < 25:
            last_reward = 10
            if swap == 1:
                goal_x = 252
                goal_y = 75
                swap = 2
            elif swap == 2:
                goal_x = 495
                goal_y = 443
                swap = 3
            else:
                goal_x = 1148
                goal_y = 523
                swap = 1

        last_distance = distance

        #car1
        if 0 <= int(self.car1.x) < sand.shape[0] and 0 <= int(self.car1.y) < sand.shape[1]:
            if sand[int(self.car1.x),int(self.car1.y)] > 0:
                self.car1.velocity = Vector(0.5, 0).rotate(self.car1.angle)
                self.penalty_counter_car1 += 1  # Tăng số lần chạm biên cho car1
                last_reward1 = -10
                self.penalty_counter_car1 += 1

               

                if self.penalty_counter_car1 >= self.max_penalty:
                    # print("Car-1 chạm biên quá nhiều lần - Đưa về vị trí trung tâm")
                    self.car1.center = self.center   # Đưa car1 về vị trí trung tâm
                    self.car1.velocity = Vector(0, 0)  # Đặt lại vận tốc của xe

                    self.car1.angle += random.choice([45, 90, -45])  # Quay xe 45 độ để rời khỏi tường
                    if self.car1.angle >= 360:  # Đảm bảo góc luôn trong khoảng [0, 360]
                         self.car1.angle -= 360

                    self.penalty_counter_car1 = 0  
            else: # otherwise
                self.car1.velocity = Vector(2, 0).rotate(self.car1.angle)
                last_reward1 = 3
                # print("car-2", 0, goal_x1, goal_y1, distance1, int(self.car1.x),int(self.car1.y), im.read_pixel(int(self.car1.x),int(self.car1.y)))
                self.penalty_counter_car1 = 0

                if distance1 < last_distance1:
                    last_reward1 += 0.2 + (1 / distance1 + 0.01) * 10
                else:
                    last_reward1 -= -0.2 * (distance1 - last_distance1)
        else:
            # Nếu xe car1 vượt khỏi giới hạn của mảng sand, đặt lại vị trí và áp dụng hình phạt nặng
            self.car1.center = self.center  # Đưa xe về vị trí trung tâm
            self.car1.velocity = Vector(0, 0)  # Đặt lại vận tốc của xe
            last_reward1 = -10  # Hình phạt lớn hơn khi vượt ra ngoài giới hạn

        if distance1 < 25:
            last_reward1 = 10
            if swap1 == 1:
                goal_x1 = 252
                goal_y1 = 75
                swap1 = 2
            elif swap1 == 2:
                goal_x1 = 495
                goal_y1 = 443
                swap1 = 3
            else:
                goal_x1 = 1148
                goal_y1 = 523
                swap1 = 1
        last_distance1 = distance1

        if brain.score() > 100 or len(scores) % 1000 == 0:
            brain.save()
            brain1.save()
            print("Model saved for both cars.")

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/Mysand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        Clock.schedule_once(self.load, 300)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        brain.load()
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        brain1.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
