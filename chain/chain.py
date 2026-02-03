"""
Processing chain module for audio signal processing.
This module defines a ProcessingChain class that manages a sequence of processing modules.
Modular design allows for flexible addition, removal, and execution of processing steps on audio data.
Ultimately varying the outcome based on the modules included in the chain.
"""

class ProcessingChain:
    def __init__(self, modules = None):
        self.modules = modules or []

        def add(self, module):
            self.modules.append(module)

        def insert(self, index, module):
            self.modules.insert(index, module)

        def remove(self, index):
            self.modules.pop(index)

        def initiate(self, audio, context):
            for module in self.modules:
                audio = module.process(audio, context)
            return audio