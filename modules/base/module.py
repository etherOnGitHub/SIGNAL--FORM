"""
Base module class for audio processing modules.
This module defines a base class for all audio processing modules in the system.
"""

class BaseModule:
    name = "BaseModule"

    # return a string representation of the module
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    # Abstract process method to be implemented by subclasses
    def process(self, audio, context):
        raise NotImplementedError("Process method must be implemented by subclasses.")
    