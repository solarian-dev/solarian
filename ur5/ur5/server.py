from proto.control.server import RelayServer
from proto.common.configs import ProjectConfig


class UR5Server(RelayServer):
    def __init__(self, cfg: ProjectConfig):
        super().__init__(cfg)
    