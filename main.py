import numpy as np


class Task:
    def __init__(self, type, deadline, arrival_time):
        self.arrival_time = arrival_time
        self.process_time = -1
        self.deadline = deadline
        self.q1_time = -1
        self.q2_time = -1
        self.type = type
        self.core = None


class Core:
    def __init__(self):
        self.idle = True
        self.rate = 0
        self.current_task = None
        self.current_start_time = -1


class Server:
    def __init__(self, cores: list[Core]):
        self.cores = cores
        self.queue: list[Task] = []

    def assign_task(self):
        pass


class Scheduler:
    def __init__(self):
        pass
