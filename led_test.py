import serial
import time

arduino = serial.Serial('/dev/cu.usbserial-1120', 9600)
time.sleep(2)  # let Arduino reset

while True:
    value = input("Enter 1 or 0: ")

    if value == '1':
        arduino.write(b'1')
    elif value == '0':
        arduino.write(b'0')