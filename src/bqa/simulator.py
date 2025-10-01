from collections.abc import Iterable
from multimethod import multimethod
from abc import ABC, abstractmethod

class Tensor(ABC):

    @property
    @abstractmethod
    def raw_tensor(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}({self.raw_tensor.__str__()})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.raw_tensor.__repr__()})"

# Task abstraction

def make_empty_task():
    raise NotImplementedError()

def add_interaction(task, lhs_node, rhs_node, ampl = 1.):
    raise NotImplementedError()

def add_local_field(taks, node, ampl = 0.):
    raise NotImplementedError()

def compile_to_context(
        task,
        max_node_dimension,
        max_bond_dimension,
        tensor_type,
):
    raise NotImplementedError()

# State abstraction

@multimethod
def make_product_state(context: dict): # type: ignore
    raise NotImplementedError()

@multimethod
def make_product_state(local_state: Tensor, context: dict): # type: ignore # noqa: F811
    raise NotImplementedError()

@multimethod
def make_product_state(local_states: Iterable, context: dict): # type: ignore # noqa: F811
    raise NotImplementedError()

@multimethod
def make_product_state(redefine_default: dict, context: dict): # noqa: F811
    raise NotImplementedError()

# --------------------------

def run(
        state,
        circuit_schedule,
        context,
):
    raise NotImplementedError()

def measure(state, context):
    raise NotImplementedError()

def compute_density(state, context):
    raise NotImplementedError()
