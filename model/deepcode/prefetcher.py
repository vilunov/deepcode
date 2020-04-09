import threading
from queue import Queue


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, buffer_size: int = 8):
        super().__init__()
        self.queue = Queue(buffer_size)
        self.generator = generator
        self.daemon = True
        self.start()

    def __iter__(self):
        return self

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __len__(self):
        return len(self.generator)

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)
