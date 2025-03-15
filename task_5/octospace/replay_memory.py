from random import sample
import queue

class Replay_Memory():

    def __init__(self):
        self.storage = queue.Queue()

    def store(self, data):
        self.storage.put(data)
    
    def get_sample(self, batch_size):
        return sample(self.storage, batch_size)
