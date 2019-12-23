import numpy as np
import os, sys
import utils

PATH = "C:\\Users\\hosse\\Desktop\\Simulation_Course_Project"
TASK_COUNT = 10000
TYPE_PARAM = 0.1

import queue

class Task:
    def __init__(self, task_type, deadline, arrival_time):
        self.arrival_time = arrival_time
        self.process_time = -1
        self.deadline = deadline
        self.q1_time = -1
        self.q2_time = -1
        self.type = task_type
        self.core = None


class Core:
    def __init__(self, mu):
        self.idle = True
        self.mu = mu  # pdf server
        self.current_task = None
        self.current_start_time = -1  # ???
        self.current_exec_time = -1  # generated with mu

    def insert_new_task(self, task: Task):
        if not self.idle:
            print("are you retarded?")
        else:
            self.idle = False
            self.current_task = task
            self.current_start_time = time
            self.current_exec_time = utils.generate_exponential(self.mu)
            self.current_task.process_time = self.current_exec_time
            self.current_task.core = self

    def update(self): # check if its finishable
        if time == self.current_exec_time + self.current_start_time:
            self.current_task = None
            self.idle = True
            self.current_exec_time = -1
            self.current_start_time = -1


class Server:
    def __init__(self, cores: list[Core]):
        self.cores = cores
        self.queue: list[Task] = []

    def handle_queue(self):
        inserted_count = 0
        for task in self.queue:
            if time - task.arrival_time > task.deadline:
                continue
            for core in self.cores:
                if core.idle:
                    core.insert_new_task(task)
                    inserted_count += 1
            if task.core is None:
                break
        self.queue = self.queue[inserted_count:]


class Scheduler:
    def __init__(self, mu, servers: list[Server]):
        self.servers: list[Server] = []
        self.rate = mu
        self.queue1: list[Task] = []
        self.queue2: list[Task] = []

    def handle_queue(self):  # number of tasks to send in and send them TODO check deadlines
        pass

    def update(self, tasks_to_insert: list[Task]):
        self.insert_new_tasks(tasks_to_insert)
        self.handle_queue()

    def update_servers(self):
        for server in self.servers:
            server.handle_queue()

    def update_cores(self):
        for server in self.servers:
            for core in server.cores:
                core.update()

    def insert_new_tasks(self, tasks_to_insert):
        for task in tasks_to_insert:
            if task.type == 1:
                self.queue1.append(task)
            else:
                self.queue2.append(task)


def initialize_servers(file_name: str):
    servers = []
    with open(os.path.join(PATH, file_name)) as f:
        lines = f.readlines()
        first_params = map(float, lines[0].split(" "))
        M = int(first_params[0])
        arrival_lamda = first_params[1]
        deadline_alpha = first_params[2]
        scheduler_mu = first_params[3]
        for line in lines[1:]:
            params = map(float, line.split(" "))
            cores = [Core(mu) for mu in params[1:]]
            servers.append(Server(cores))
    scheduler = Scheduler(scheduler_mu, servers)
    return scheduler, arrival_lamda, deadline_alpha


def Task_Generator(arrival_lamda, deadline_alpha):
    tasks: list[Task] = []
    arrivals = []
    t = 0
    for i in range(TASK_COUNT):
        type = utils.generate_type(TYPE_PARAM)
        deadline = round(utils.generate_exponential(deadline_alpha))
        inter_arrival = round(utils.generate_exponential(1 / arrival_lamda))
        arrival = t + inter_arrival
        temp_task = Task(type, deadline, arrival)
        tasks.append(temp_task)
        t += inter_arrival
    return tasks


def insertable_tasks(tasks: list[Task], current_time, current_task_pointer):
    tasks_to_add = []
    i = current_task_pointer
    while True:
        if tasks[i].arrival_time == current_time:
            tasks_to_add.append(tasks[i])
            i += 1
        elif tasks[i].arrival_time > current_time:
            break
    return tasks_to_add


def start(scheduler: Scheduler, tasks: list[Task]):
    global time
    time = 0
    current_task_pointer = 0
    while True:
        scheduler.update_cores()
        scheduler.update_servers()
        tasks_to_insert = insertable_tasks(tasks, time, current_task_pointer)
        scheduler.update(tasks_to_insert)
        scheduler.update_servers()
        scheduler.update_cores()
        time += 1
