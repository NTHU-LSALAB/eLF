import _elf


class Controller:
    def __init__(self, address):
        self._ctrl = _elf.create_controller()
        self._ectrl = _elf.export_controller(self._ctrl, address)

    @property
    def listening_port(self):
        return self._ectrl.listening_port()

    def stop(self):
        self._ectrl.stop()
        self._ctrl.stop()


class Worker(_elf.Worker):
    def __init__(self, controller_address):
        self._ctrl = _elf.connect_controller(controller_address)
        super().__init__(self._ctrl)

    def shard_generator(self, list_=None, *, batch_size=1):
        while True:
            should_continue, requires_broadcast, shard_number = self.begin_batch()
            if not should_continue:
                break
            if requires_broadcast:
                self.broadcast()
            if list_ is None:
                yield shard_number
            else:
                yield [list_[shard_number * batch_size + i] for i in range(batch_size)]

    def broadcast(self):
        print('broadcast triggered')