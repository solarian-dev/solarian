from abc import ABC, abstractmethod
from typing import Any
from solarian.core.configs import ComponentConfig
# from solarian.core.joint import Joint
# from solarian.core.body import Body
from pydantic import BaseModel


class Component(BaseModel):
    """Base class for components"""
    id: int
    model: str

    
    
    # @abstractmethod
    # def connect(self):
    #     """Connect the component."""
    #     pass

    # @abstractmethod
    # def disconnect(self):
    #     """Disconnect the component."""
    #     pass

    # @abstractmethod
    # def needs_setup(self) -> bool:
    #     """Check if the component needs setup."""
        
    

    

    
