from proto.control.server import RelayServer
from proto.common.configs import ProjectConfig


class LeKiwiServer(RelayServer):
    def __init__(self, cfg: ProjectConfig):
        super().__init__(cfg)
    