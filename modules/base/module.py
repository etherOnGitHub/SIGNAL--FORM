"""
Base module class for audio processing modules.
This module defines a base class for all audio processing modules in the system.
"""

class BaseModule:
    name = "BaseModule"

    def process(self, audio, context):
        raise NotImplementedError("Process method must be implemented by subclasses.")
    