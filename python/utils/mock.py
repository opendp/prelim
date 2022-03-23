from dataclasses import dataclass
from typing import Callable


@dataclass
class MockTransformation(object):
    function: Callable
    forward_map: Callable

    def __call__(self, arg):
        return self.function(arg)

    def forward_map(self, d_in):
        return self.relation(d_in)

@dataclass
class MockMeasurement(object):
    function: Callable
    forward_map: Callable

    def __call__(self, arg):
        return self.function(arg)

    def forward_map(self, d_in):
        return self.forward_map(d_in)
