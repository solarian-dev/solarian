import msgpack
from pydantic import BaseModel
from typing import Dict, Any

class Message(BaseModel):

    robot_timestamp: int = 0
    server_timestamp: int = 0
    controller_timestamp: int = 0
    data: Dict[str, Any] = {}

    def serialize(self) -> bytes:
        """Convert observation to msgpack format for transmission"""
        return msgpack.packb(self.model_dump())

    @classmethod
    def deserialize(cls, data: bytes) -> "Message":
        """Create Message instance from msgpack data"""
        unpacked = msgpack.unpackb(data)
        return cls(**unpacked)
