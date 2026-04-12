class ComponentNotConnectedError(Exception):
    """Exception raised when the component is not connected."""

    def __init__(
        self, message="This component is not connected. Try calling `component.connect()` first."
    ):
        self.message = message
        super().__init__(self.message)


class ComponentAlreadyConnectedError(Exception):
    """Exception raised when the component is already connected."""

    def __init__(
        self,
        message="This component is already connected. Try not calling `component.connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)


class JointOutOfRangeError(Exception):
    def __init__(self, message="Joint is out of range"):
        self.message = message
        super().__init__(self.message)