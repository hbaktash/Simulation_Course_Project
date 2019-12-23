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
    def __init__(self, mu):
        self.idle = True
        self.rate = mu                  #pdf server
        self.speed = -1                 #pdf server
        self.current_task = None
        self.current_start_time = -1    #???
        
    def set_speed(self):
        pass
        
    def finish_task(self):
        pass


class Server:
    def __init__(self, cores: list[Core]):
        self.cores = cores
        self.queue: list[Task] = []

    def assign_task(self):              #set on first free core
        pass
    
    def queue_update(self):
        pass
    


class Scheduler:
    def __init__(self, mu):
        self.rate = mu
        self.speed = -1
        self.queue1: list[Task] = []
        self.queue2: list[Task] = []
        
    def set_speed(self):                # mu different for each task!!
        pass
    
    def assign_task(self):
        pass
    
def Task_Generator():
    pass                                #gen arrival(lambda) + deadline(alpha)
