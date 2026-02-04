"""
Base module class for audio processing modules.
This module defines a base class for all audio processing modules in the system.

All modules use Numpy mono (T,) audio format for input and output.
This is the standard format for audio processing in Python and is compatible with most audio libraries.

Modules that use machine learning models internally can implement their own adapters to convert between
Numpy and the required input format for the model (e.g. PyTorch tensors).
This allows us to keep the module interface consistent while still being able to leverage powerful ML models.
"""

class BaseModule:
    name = "BaseModule"

    # return a string representation of the module
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    # Abstract process method to be implemented by subclasses
    def process(self, audio, context):
        raise NotImplementedError("Process method must be implemented by subclasses.")
    