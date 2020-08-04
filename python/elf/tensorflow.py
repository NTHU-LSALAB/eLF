import os
import elf
import tensorflow as tf


op = tf.load_op_library(os.path.join(os.environ.get('ELF_LIB_DIR', '.'), 'lib_elf_tensorflow.so'))


class _Delegate:
    def __init__(self, to):
        self.to = to
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner=None):
        if owner is None:
            return instance
        obj = getattr(instance, self.to)
        return getattr(obj, self.name)


class ElasticOptimizer(tf.train.Optimizer):
    def __init__(self, worker: elf.Worker, optimizer: tf.train.Optimizer, name=None, use_locking=False):
        self._worker = worker
        self._optimizer = optimizer
        self._operators = []  # hold the reference to C++ objects
        self._broadcast_op = self._make_broadcast_op(tf.global_variables())
        self._worker.broadcast_fn = self._broadcast_fn
        if name is None:
            name = 'Elastic' + type(optimizer).__name__
        super().__init__(name=name, use_locking=use_locking)

    def commit_and_join(self):
        self._worker.commit_and_join()

    def _make_broadcast_op(self, variables):
        operations = []
        for var in variables:
            operator = self._worker.add_global_variable(var.name)
            self._operators.append(operator)
            operations.append(var.assign(op.value_operator(var, handle=operator.get_handle())))
        return tf.group(*operations)

    def _broadcast_fn(self):
        tf.get_default_session().run(self._broadcast_op)

    def compute_gradients(self, *args, **kwargs):
        gradients = self._optimizer.compute_gradients(*args, **kwargs)
        sum_grads = []
        for grad, var in gradients:
            if grad is not None:
                operator = self._worker.add_weight_variable(var.name)
                self._operators.append(operator)
                sum_ = op.value_operator(grad, handle=operator.get_handle())
                sum_grads.append((sum_, var))
            else:
                sum_grads.append((None, var))
        return sum_grads

    apply_gradients = _Delegate('_optimizer')
    get_slot = _Delegate('_optimizer')
    get_slot_names = _Delegate('_optimizer')
    variables = _Delegate('_optimizer')
