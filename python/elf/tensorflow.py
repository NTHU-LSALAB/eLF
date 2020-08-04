import elf
import tensorflow as tf


op = tf.load_op_library


class _Delegate:
    def __init__(self, to):
        self.to = to
        self.name = None

    def __set_name__(self, name):
        self.name = name

    def __get__(self, instance, owner=None):
        if owner is None:
            return instance
        obj = getattr(instance, self.to)
        return getattr(obj, self.name)


class ElasticOptimizer(tf.train.Optimizer):
    def __init__(self, worker: elf.Worker, optimizer: tf.train.Optimizer):
        self._worker = worker
        self._optimizer = optimizer
        self._operators = []
        super().__init__()

    def compute_gradients(self, *args, **kwargs):
        gradients = self._optimizer.compute_gradients(*args, **kwargs)
        sum_grads = []
        for grad, var in gradients:
            if grad is not None:
                operator = self._worker.add_weight_variable(var.name)
                sum_ = op.value_operator(grad, handle=operator.get_handle())
                sum_grads.append((sum_, var))
            else:
                sum_grads.append((None, var))
        return sum_grads

    apply_gradients = _Delegate('_optimizer')
    get_slot = _Delegate('_optimizer')
    get_slot_names = _Delegate('_optimizer')
    variables = _Delegate('_optimizer')
