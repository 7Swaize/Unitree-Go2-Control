
import struct
from dataclasses import dataclass


@dataclass
class ControllerState:
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    
    l1: float = 0.0
    l2: float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    
    a: float = 0.0
    b: float = 0.0
    x: float = 0.0
    y: float = 0.0
    
    up: float = 0.0
    down: float = 0.0
    left: float = 0.0
    right: float = 0.0
    
    select: float = 0.0
    start: float = 0.0
    f1: float = 0.0
    f3: float = 0.0

    changed: bool = False

class UnitreeRemoteControllerInputParser:
    def __init__(self) -> None:
        self._state = ControllerState()

    def _parse_buttons(self, data1: int, data2: int) -> None:
        mapping1 = {
            0: "r1", 1: "l1", 2: "start", 3: "select",
            4: "r2", 5: "l2", 6: "f1", 7: "f3"
        }
        mapping2 = {
            0: "a", 1: "b", 2: "x", 3: "y",
            4: "up", 5: "right", 6: "down", 7: "left"
        }

        for i, attr in mapping1.items():
            setattr(self._state, attr, (data1 >> i) & 1)

        for i, attr in mapping2.items():
            setattr(self._state, attr, (data2 >> i) & 1)

    def _parse_analog(self, data: bytes):
        lx_offset = 4
        self.lx = struct.unpack('<f', data[lx_offset:lx_offset + 4])[0]
        rx_offset = 8
        self.Rx = struct.unpack('<f', data[rx_offset:rx_offset + 4])[0]
        ry_offset = 12
        self.Ry = struct.unpack('<f', data[ry_offset:ry_offset + 4])[0]
        L2_offset = 16
        L2 = struct.unpack('<f', data[L2_offset:L2_offset + 4])[0] # Placeholder，unused
        ly_offset = 20
        self.Ly = struct.unpack('<f', data[ly_offset:ly_offset + 4])[0]

    # remote_data is some type that is an array of 8-bit-seqs
    def parse(self, remote_data) -> ControllerState:
        self._previous_state = ControllerState(**self._state.__dict__)
        
        self._parse_analog(remote_data)
        self._parse_buttons(remote_data[2], remote_data[3])
    
        return self._state
