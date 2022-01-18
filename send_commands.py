#!/usr/bin/env python2
from __future__ import print_function

import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey
import random

def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def main():
    #signal.signal(signal.SIGINT, terminate_program)

    # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
    rob = robobo.SimulationRobobo().connect(address='192.168.1.133', port=19997)

    rob.play_simulation()

    # Following code moves the robot
    for i in range(100):
            #print("robobo is at {}".format(rob.position()))
            r, l = random.choice([100,70,66]), random.choice([5,18,22])
            rob.move(r, l, 100)
            #print(rob.read_irs())
            print(rob.get_image_front())
    rob.stop_world()
    return
    print("robobo is at {}".format(rob.position()))
    rob.sleep(1)
    # Following code moves the phone stand
    rob.set_phone_pan(343, 100)
    rob.set_phone_tilt(109, 100)
    time.sleep(1)
    rob.set_phone_pan(11, 100)
    rob.set_phone_tilt(26, 100)

    # Following code makes the robot talk and be emotional
    rob.set_emotion('happy')
    rob.talk('Hi, my name is Robobo')
    rob.sleep(1)
    rob.set_emotion('sad')

    # Following code gets an image from the camera
    image = rob.get_image_front()
    # IMPORTANT! `image` returned by the simulator is BGR, not RGB
    cv2.imwrite("test_pictures.png",image)

    time.sleep(0.1)

    # IR reading
    for i in range(1000000):
        print("ROB Irs: {}".format(np.log(np.array(rob.read_irs()))/10))
        time.sleep(0.1)

    # pause the simulation and read the collected food
    rob.pause_simulation()
    
    # Stopping the simualtion resets the environment
    rob.stop_world()


if __name__ == "__main__":
    main()
