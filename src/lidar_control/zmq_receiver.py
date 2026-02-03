import zmq
import threading
import numpy as np

from src.lidar_control.callback_dispatcher import CallbackDispatcher


# pip install pyzmq

class ZMQReceiver(threading.Thread):
    def __init__(self, endpoint: str, dispatcher: CallbackDispatcher):
        super().__init__(daemon=True)
        self.dispatcher = dispatcher
        self.ctx = zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.connect(endpoint)

        self.socket.setsockopt_string(zmq.SUBSCRIBE, "decoded_topic")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "filtered_topic")

        self._running = threading.Event()
        self._running.set()

    def run(self) -> None:
        self._running = True
        
        while self._running:
            try:
                topic = self.socket.recv_string()
                md = self.socket.recv_json()
                buf = self.socket.recv(copy=False)

                arr = np.frombuffer(buf, dtype=np.dtype(md["dtype"]))
                arr = arr.reshape(md["shape"])

                self.dispatcher.emit(topic, md["stamp_ns"], arr)

            except Exception as e:
                print(f"Failed too receive, process, or dispatch LIDAR from ZMQ publisher: {e}")


    def shutdown(self) -> None:
        self._running.clear()
        self.socket.close()