from tcp_client import run_tcp_client
from servo_controller import setup_servo, move_servo

def main():
    run_tcp_client()

if __name__ == "__main__":
    main()
    #servo = setup_servo()
    #while True:
    #    move_servo(servo)
