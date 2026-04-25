import sys
import time
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize



if len(sys.argv) < 2:
    ChannelFactoryInitialize(1, "lo")
else:
    ChannelFactoryInitialize(0, sys.argv[1])

sp = SportClient()
sp.Init()
sp.SetTimeout(1.0)

start = time.perf_counter()
sp.Move(vx=1, vy=0, vyaw=0)
end = time.perf_counter()

print(f"Execution time: {end - start:.6f} seconds.")