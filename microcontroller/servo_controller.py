"""
A simple example that sweeps a Servo back-and-forth
Requires the micropython-servo library - https://pypi.org/project/micropython-servo/
"""

import time
from servo import Servo

def setup_servo():
    servo = Servo(pin_id=16)
    print("[SERVO_CONTROLLER] Servo correctamente configurado")
    return servo

def move_servo(my_servo):
    my_servo.write(60)
    time.sleep_ms(1000)
    my_servo.write(180)
    time.sleep_ms(1000)
