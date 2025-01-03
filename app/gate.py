import biostarPython as g
from biostarPython.service import connect_pb2, door_pb2

from app.config import (
    CONTROLLER_CA_PATH,
    CONTROLLER_IP,
    CONTROLLER_PORT,
    GATE_ID,
)


class GateControl:
    def __init__(
        self,
        gateway_ip=CONTROLLER_IP,
        gateway_port=CONTROLLER_PORT,
        ca_file=CONTROLLER_CA_PATH,
    ):
        self.gateway = g.GatewayClient(gateway_ip, gateway_port, ca_file)
        self.channel = self.gateway.getChannel()
        self.connect_svc = g.ConnectSvc(self.channel)
        self.door_svc = g.DoorSvc(self.channel)
        devices = self.connect_svc.searchDevice(300)
        if not devices:
            raise RuntimeError("No device found")
        conn_info = connect_pb2.ConnectInfo(
            IPAddr=devices[0].IPAddr, port=devices[0].port, useSSL=False
        )
        self.device_id = self.connect_svc.connect(conn_info)

    def unlock(self):
        self.door_svc.unlock(self.device_id, [GATE_ID], door_pb2.OPERATOR)
        print("🔓 Gate unlocked")

    def lock(self):
        self.door_svc.lock(self.device_id, [GATE_ID], door_pb2.OPERATOR)
        print("🔒 Gate locked")
