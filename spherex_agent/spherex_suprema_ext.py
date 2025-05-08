'''
This Script implements the API to control the Sphere X Extension Hardware.
It will allow the user to open/close Gate Relays and monitor them if needed.

'''
'''
This Script implements the API to control the Sphere X Extension Hardware.
It will allow the user to open/close Gate Relays and monitor them if needed.
'''

import serial
import serial.tools.list_ports
from time import sleep

class Barrier:
    '''
    Abstraction for one barrier

    To Control a barrier, send the Code:
    "Xi<relay number>"

    The onboard STC Microcontroller will parse the code and execute command.
    '''
    def __init__(self, gate, open_relay_ind, close_relay_ind):
        '''
        Initialize barrier with open and close relay indices
        '''
        self.gate = gate
        self.open_relay_ind = open_relay_ind
        self.close_relay_ind = close_relay_ind

    def open(self):
        '''
        opens barrier by toggling open relay
        '''
        command = f"Xi{self.open_relay_ind}"
        self.gate.send_command(command)

    def close(self):
        '''
        closes barrier by toggling close relay
        '''
        command = f"Xi{self.close_relay_ind}"
        self.gate.send_command(command)

class Gate:
    '''
    Abstraction to control One Gate:
    Each Gate has 4 relays.
    Relay index 0: Opens Entry barrier
    Relay index 1: Closes Entry barrier
    Relay index 2: Opens Exit barrier
    Relay index 3: Closes Exit barrier
    '''
    def __init__(self, port=None, baudrate=115200, timeout=1):
        '''
        1- Establish connection to STC Microcontroller
        2- Test Connection is Stable
        '''
        self.serial_conn = None
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        
        # Initialize connection
        self.connect()
        
        # Test connection
        self.test_connection()
        
        # Create Barrier instances
        self.entry_barrier = Barrier(self, 0, 1)
        self.exit_barrier = Barrier(self, 2, 3)

    def find_cp2102(self):
        '''
        Find CP2102 converter in connected USB devices
        Returns the port if found, None otherwise
        '''
        print("Looking for CP2102 using Vendor ID and Product ID..")

        cp210_vid = 0x10C4  # CP2102 Vendor ID
        cp210_pid = 0xEA60  # CP2102 Product ID
        
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if port.vid == cp210_vid and port.pid == cp210_pid:
                print(f"Found CP2102 Device at {port.device}\n")
                return port.device
        return None

    def connect(self):
        '''
        Establish serial connection to the microcontroller
        '''
        if self.port is None:
            self.port = self.find_cp2102()
            if self.port is None:
                raise Exception("CP2102 device not found. Please check connection.")
        
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            # Wait for connection to stabilize
            sleep(2)
        except serial.SerialException as e:
            raise Exception(f"Failed to connect to {self.port}: {str(e)}")

    def test_connection(self):
        '''
        Test if the connection is working by sending a newline and expecting 'Enter Received..'
        '''
        try:
            # Sending enter to STC should respond with "Enter Received.."
            response = self.send_command('')
            print(response)
            if 'Enter Received' not in response:
                raise Exception(f"Unexpected response from controller: {response}")
        except Exception as e:
            raise Exception(f"Connection test failed: {str(e)}")

    def send_command(self, command) -> str:
        '''
        Send a command to the microcontroller and wait for response
        '''
        if not self.serial_conn or not self.serial_conn.is_open:
            raise Exception("Serial connection not established")
        
        try:
            self.serial_conn.write(f"{command}\n".encode())
            return self.serial_conn.readline().decode().strip()
        except serial.SerialException as e:
            raise Exception(f"Error sending command: {str(e)}")

    def close_connection(self):
        '''
        Close the serial connection
        '''
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

    def __del__(self):
        '''
        Destructor to ensure serial connection is closed
        '''
        self.close_connection()


# Example usage
if __name__ == "__main__":
    try:
        # Create gate instance (automatically connects)
        gate = Gate()
        
        print("Connected to gate controller")
        
        # Control entry barrier
        print("Opening entry barrier")
        gate.entry_barrier.open()
        sleep(2)  # Wait for operation to complete
        
        print("Closing entry barrier")
        gate.entry_barrier.close()
        sleep(2)
        
        # Control exit barrier
        print("Opening exit barrier")
        gate.exit_barrier.open()
        sleep(2)
        
        print("Closing exit barrier")
        gate.exit_barrier.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")