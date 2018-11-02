from math import sqrt
import cv2
import time
from threading import *
from datetime import datetime

# ML Classifier to define hands - Use definite path of Haar Classifier
hand_cascade = cv2.CascadeClassifier('E:\HandGestureRecognition\Gest.xml')

# set array if tuples that hold x and y values of the location of the hand and time
posx = []
posy = []
time_array = []

# rate at which to check if gesture is done
refresh = 0

# how long between gestures determines if we're done swiping (ms)
sleep = 1500  # 1.5 seconds


class perpetualTimer():

    def __init__(self, t, hFunction):
        self.t = t
        self.hFunction = hFunction
        self.thread = Timer(self.t, self.handle_function)

    def handle_function(self):
        self.hFunction()
        self.thread = Timer(self.t, self.handle_function)
        self.thread.start()

    def start(self):
        self.thread.start()

    def cancel(self):
        self.thread.cancel()


# returns the elapsed milliseconds
def millis(past_time):
    dt = datetime.now() - past_time
    ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
    return ms


def increment_rate():
    global refresh
    if refresh >= 3:
        refresh = 0
    refresh += 1
    print(refresh)


def standard_deviation(lst):
    """Calculates the standard deviation for a list of numbers."""

    num_items = len(lst)
    mean = sum(lst) / num_items
    differences = [x - mean for x in lst]
    sq_differences = [d ** 2 for d in differences]
    ssd = sum(sq_differences)
    variance = ssd / num_items
    return sqrt(variance)


def check_gesture(x, y, array_time, sleep1, refresh_rate):
    """ runs periodically, at time refresh_rate, to check the position arrays to see if a gesture was performed. """
    # refresh is the rate at which we want to check it
    # sleep - time between gestures to determine if we stopped
    rate = refresh_rate
    locx = x
    locy = y
    time_array1 = array_time
    length = len(time_array1)
    sd_x = standard_deviation(x)
    sd_y = standard_deviation(y)
    rangex = locx[length - 1] - locx[0]
    rangey = locy[length - 1] - locy[0]

    # every 2 seconds, check the arrays
    if rate == 2:
        # if the time between the last gesture movement is too great, analyze the gesture b/c gesture is done
        last_time = time_array1[length - 1]
        elapsed_time = millis(last_time)
        if elapsed_time >= sleep1:
            # check if the y standard deviation is large, if not, it's horizontal movement
            if sd_y <= 40:
                if rangex > 0:
                    print("you're swiping right")
                else:
                    print("you're swiping left")
            else:
                if rangey > 0:
                    print("you're swiping down")
                else:
                    print("you're swiping up")
            posx.clear()
            posy.clear()
            time_array.clear()


def motion():
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hands = hand_cascade.detectMultiScale(gray, 1.3, 5)

        # track right hand by using original image
        for (x, y, w, h) in hands:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            centerx = x + int(w / 2)
            centery = y + int(h / 2)
            current_millis = datetime.now()
            cv2.circle(img, (centerx, centery), int(h / 2), (0, 255, 0), 2)
            cv2.circle(img, (centerx, centery), 5, (0, 255, 0), -1)
            posx.append(centerx)
            posy.append(centery)
            time_array.append(current_millis)

        # Check if arrays have values otherwise, things stop working
        if len(posx) != 0:
            check_gesture(posx, posy, time_array, sleep, refresh_rate=refresh)

        cv2.imshow('img', img)

        # press escape to stop tracking and end program
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
        time.sleep(50 / 1000.0)


"""
To DO:

- CHECK FOR HORIZONTAL AND VERTICAL MOVEMENT IN ORDER TO DETERMINE DIAGONAL MOVEMENT 

- May be possible to use both hands if we define left hand just for swiping left and right for swiping right
  have two separate arrays for each hand and we would check the size of array created and compare them
  the bigger array would be used as the one to determine motion, therefore checking which hand we used
    - or just retrain the classifier to be more accurate 

"""


def start():
    thread1 = Thread(target=motion)
    thread1.start()
    t = perpetualTimer(1, increment_rate)
    t.start()


start()
