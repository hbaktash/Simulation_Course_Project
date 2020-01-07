import os
import math
import numpy as np
from typing import TypeVar, List

import utils

PATH = "C:\\Users\\hosse\\Desktop\\Simulation_Course_Project"
TASK_COUNT = 600000
TYPE_PARAM = 0.1
MAX_TASKS = 50000000
IGNORED_TASKS = 1000
DELTA_T = 0.2

time = 0
out_tasks = 0
update_stats = False


class Data:
    def __init__(self, name: str):
        self.name = name
        self.n: int = 0
        self.mean: float = 0
        self.std: float = 0
        self.last_data = None

    def update(self, new_data: float):
        global update_stats
        if update_stats:
            self.mean = ((self.mean * self.n) + new_data) / (self.n + 1)
            if self.n > 2:
                self.std = math.sqrt(((self.std ** 2) * (self.n - 1) + (new_data - self.mean) ** 2) / self.n)
            elif self.n == 2:
                self.std = math.sqrt((new_data - self.mean) ** 2 + (self.last_data - self.mean) ** 2)
            self.n += 1
            self.last_data = new_data

    def accuracy(self):  # 95 %
        if self.n >= 2:
            if self.mean == 0:
                return 0
            return 1.96 * self.std / (math.sqrt(self.n) * self.mean)
        return 1

    def sum(self):
        return self.n * self.mean


system_time_stat_1 = Data("sys time 1")
system_time_stat_2 = Data("sys time 2")
queue_time_stat_1 = Data("q time 1")
queue_time_stat_2 = Data("q time 2")
missed_deadlines_stat_1 = Data("missed deadlines 1")
missed_deadlines_stat_2 = Data("missed deadlines 2")


class Task:
    def __init__(self, task_type, deadline, arrival_time, id=-1):
        self.id = id
        self.arrival_time = arrival_time
        self.server_process_time = -1
        self.scheduler_process_time = -1
        self.deadline = deadline
        self.q1_time = -1
        self.q2_time = -1
        self.type = task_type
        self.core = None

    def print_task_data(self):
        print(
            "task {0}\n    arrival time: {1}\n    deadline: {2}\n    scheduler process: {3}\n    server process:{4}\n    q1: {5}\n    q2: {6}".format(
                self.id, self.arrival_time, self.deadline, self.scheduler_process_time, self.server_process_time,
                self.q1_time,
                self.q2_time))


class Core:
    def __init__(self, mu):
        self.idle = True
        self.mu = mu
        self.current_task: Task = None
        self.current_start_time = -1
        self.current_exec_time = -1  # generated with mu

    def insert_new_task(self, task: Task):
        if not self.idle:
            print("are you retarded?")
        else:
            self.idle = False
            self.current_task = task
            self.current_start_time = time
            self.current_exec_time = round(utils.generate_exponential(self.mu))
            self.current_task.server_process_time = self.current_exec_time
            self.current_task.core = self

    def update(self):  # check if its finishable
        global out_tasks, system_time_stat_1, system_time_stat_2, missed_deadlines_stat_1, missed_deadlines_stat_2
        if self.idle:
            return
        if time >= self.current_exec_time + self.current_start_time:
            if self.current_task.type == 1:
                system_time_stat_1.update(self.current_task.q1_time +
                                          self.current_task.q2_time +
                                          self.current_task.server_process_time +
                                          self.current_task.scheduler_process_time)
                missed_deadlines_stat_1.update(0)
            else:
                system_time_stat_2.update(self.current_task.q1_time +
                                          self.current_task.q2_time +
                                          self.current_task.server_process_time +
                                          self.current_task.scheduler_process_time)
                missed_deadlines_stat_2.update(0)
            self.current_task = None
            self.idle = True
            self.current_exec_time = -1
            self.current_start_time = -1
            out_tasks += 1


class Server:
    def __init__(self, cores: List[Core], id: int):
        self.id = id
        self.cores = cores
        self.queue: List[Task] = []
        self.queue_length_stat = Data("server {} queue len".format(self.id))

    def handle_queue(self):
        global out_tasks, queue_time_stat_1, queue_time_stat_2, missed_deadlines_stat_1, missed_deadlines_stat_2, system_time_stat_1, system_time_stat_2
        inserted_count = 0
        for task in self.queue:
            if time - task.arrival_time > task.deadline:
                inserted_count += 1
                out_tasks += 1
                task.q2_time = task.deadline - task.q1_time - task.scheduler_process_time
                if task.type == 1:
                    queue_time_stat_1.update(task.q1_time + task.q2_time)
                    missed_deadlines_stat_1.update(1)
                    system_time_stat_1.update(task.q1_time + task.q2_time + task.scheduler_process_time)
                else:
                    queue_time_stat_2.update(task.q1_time + task.q2_time)
                    missed_deadlines_stat_2.update(1)
                    system_time_stat_2.update(task.q1_time + task.q2_time + task.scheduler_process_time)
                continue
            for core in self.cores:
                if core.idle:
                    task.q2_time = time - task.arrival_time - task.q1_time - task.scheduler_process_time
                    core.insert_new_task(task)
                    inserted_count += 1
                    if task.type == 1:
                        queue_time_stat_1.update(task.q1_time + task.q2_time)
                    else:
                        queue_time_stat_2.update(task.q1_time + task.q2_time)
            if task.core is None:
                break
        self.queue = self.queue[inserted_count:]
        self.queue_length_stat.update(len(self.queue))


class Scheduler:
    def __init__(self, mu, servers: List[Server]):
        self.servers: List[Server] = servers
        self.rate = mu
        self.idle = True
        self.current_task: Task = None
        self.next_free_time = 0
        self.queue1: List[Task] = []
        self.queue2: List[Task] = []
        self.queue_length_stat = Data("scheduler queue len")

    def handle_queue(self):  # number of tasks to send in and send them
        global out_tasks, queue_time_stat_1, queue_time_stat_2, missed_deadlines_stat_1, missed_deadlines_stat_2, system_time_stat_1, system_time_stat_2
        if time >= self.next_free_time:
            if len(self.queue1) != 0:
                current_task: Task = self.queue1[0]
                self.queue1 = self.queue1[1:]
            elif len(self.queue2) != 0:
                current_task: Task = self.queue2[0]
                self.queue2 = self.queue2[1:]
            else:
                return
            service_time = (utils.generate_exponential(self.rate))
            # print("Service time:", service_time)
            self.next_free_time = time + service_time
            if time - current_task.arrival_time > current_task.deadline:
                out_tasks += 1
                current_task.q1_time = current_task.deadline
                self.handle_queue()  # pass the passed deadline
                if current_task.type == 1:
                    queue_time_stat_1.update(current_task.q1_time)
                    missed_deadlines_stat_1.update(1)
                    system_time_stat_1.update(current_task.q1_time)
                else:
                    queue_time_stat_2.update(current_task.q1_time)
                    missed_deadlines_stat_2.update(1)
                    system_time_stat_2.update(current_task.q1_time)
            elif time - current_task.arrival_time + service_time > current_task.deadline:
                out_tasks += 1
                current_task.q1_time = time - current_task.arrival_time
                current_task.scheduler_process_time = current_task.deadline - current_task.q1_time
                if current_task.type == 1:
                    queue_time_stat_1.update(current_task.q1_time)
                    missed_deadlines_stat_1.update(1)
                    system_time_stat_1.update(current_task.deadline)
                else:
                    queue_time_stat_2.update(current_task.q1_time)
                    missed_deadlines_stat_2.update(1)
                    system_time_stat_2.update(current_task.deadline)
                self.next_free_time = time + current_task.deadline
                self.current_task = None
            else:
                current_task.q1_time = time - current_task.arrival_time
                current_task.scheduler_process_time = service_time
                self.current_task = current_task
                self.queue_length_stat.update(len(self.queue1) + len(self.queue2))

    def assign_task(self):
        self.servers.sort(key=lambda x: len(x.queue))
        min_length = len(self.servers[0].queue)
        min_server_lenghts = []
        for server in self.servers:
            if len(server.queue) == min_length:
                min_server_lenghts.append(server)
            else:
                break
        rand_num = np.random.randint(0, len(min_server_lenghts))
        # print("rand:", rand_num)
        self.servers[rand_num].queue.append(self.current_task)

    def update(self, tasks_to_insert: List[Task]):
        if self.current_task is None:
            pass
        else:
            if time >= self.next_free_time:
                self.assign_task()
                self.current_task = None

        self.insert_new_tasks(tasks_to_insert)
        self.handle_queue()

        if self.current_task is None:
            pass
        else:
            if time >= self.next_free_time:
                self.assign_task()
                self.current_task = None

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

    def all_accurate(self):
        global queue_time_stat_1, queue_time_stat_2, missed_deadlines_stat_1, missed_deadlines_stat_2, system_time_stat_1, system_time_stat_2
        count = 7 + len(self.servers)
        if self.queue_length_stat.n < 2:
            return False
        if self.queue_length_stat.accuracy() > 0.05:
            return False
        elif system_time_stat_1.accuracy() > 0.05:
            return False
        elif system_time_stat_2.accuracy() > 0.05:
            return False
        elif missed_deadlines_stat_1.accuracy() > 0.05:
            return False
        elif missed_deadlines_stat_2.accuracy() > 0.05:
            return False
        elif queue_time_stat_1.accuracy() > 0.05:
            return False
        elif queue_time_stat_2.accuracy() > 0.05:
            return False
        for server in self.servers:
            if (server.queue_length_stat.n < 2) or (server.queue_length_stat.accuracy() > 0.05):
                return False
        return True


def initialize_servers(file_name: str):
    servers = []
    with open(os.path.join(PATH, file_name)) as f:
        lines = f.readlines()
        first_params = list(map(float, lines[0].split(" ")))
        arrival_lamda = first_params[1]
        deadline_alpha = first_params[2]
        scheduler_mu = first_params[3]
        j = 0
        for line in lines[1:]:
            params = list(map(float, line.split(" ")))
            cores = [Core(mu) for mu in params[1:]]
            servers.append(Server(cores, j))
            j += 1
    scheduler = Scheduler(scheduler_mu, servers)
    return scheduler, arrival_lamda, deadline_alpha


def task_generator(arrival_lamda, deadline_alpha, task_count=TASK_COUNT):
    tasks: List[Task] = []
    t = 0
    for i in range(task_count):
        type = utils.generate_type(TYPE_PARAM)
        deadline = (utils.generate_exponential(1 / deadline_alpha))
        inter_arrival = (utils.generate_exponential(1 / arrival_lamda))
        arrival = t + inter_arrival
        temp_task = Task(type, deadline, arrival, id=i)
        tasks.append(temp_task)
        t += inter_arrival
    return tasks


def insertable_tasks(tasks: List[Task], current_time, current_task_pointer):
    tasks_to_add = []
    i = current_task_pointer
    while i < len(tasks):
        if tasks[i].arrival_time <= current_time:
            tasks_to_add.append(tasks[i])
            i += 1
        elif tasks[i].arrival_time > current_time:
            break
    return tasks_to_add


def start(scheduler: Scheduler, tasks: List[Task]):
    global time, update_stats, out_tasks
    current_task_pointer = 0
    while scheduler.all_accurate() or (out_tasks < MAX_TASKS and out_tasks < len(tasks)):
        scheduler.update_cores()
        scheduler.update_servers()
        tasks_to_insert = insertable_tasks(tasks, time, current_task_pointer)
        current_task_pointer += len(tasks_to_insert)
        scheduler.update(tasks_to_insert)
        scheduler.update_servers()
        scheduler.update_cores()
        print("outs: ", out_tasks)
        if out_tasks >= IGNORED_TASKS:
            update_stats = True
        time += DELTA_T


def print_info(tasks: List[Task]):
    i = 0
    for task in tasks:
        task.print_task_data()
        i += 1


def main():
    global queue_time_stat_1, queue_time_stat_2, missed_deadlines_stat_1, missed_deadlines_stat_2, system_time_stat_1, system_time_stat_2
    scheduler, arrival_lamda, deadline_alpha = initialize_servers("input.txt")
    tasks = task_generator(arrival_lamda, deadline_alpha, task_count=TASK_COUNT)
    start(scheduler, tasks)
    # print_info(tasks)
    print("\n\n\n###Statistics###")
    print("checked tasks:", out_tasks)
    print("q time 1:", queue_time_stat_1.mean, " with accuracy ", queue_time_stat_1.accuracy())
    print("q time 2:", queue_time_stat_2.mean, " with accuracy ", queue_time_stat_2.accuracy())
    print("---total q time:",
          (queue_time_stat_2.sum() + queue_time_stat_1.sum()) / (queue_time_stat_1.n + queue_time_stat_2.n))
    print("sys time 1:", system_time_stat_1.mean, " with accuracy ", system_time_stat_1.accuracy())
    print("sys time 2:", system_time_stat_2.mean, " with accuracy ", system_time_stat_2.accuracy())
    print("---total sys time:",
          (system_time_stat_1.sum() + system_time_stat_2.sum()) / (system_time_stat_1.n + system_time_stat_2.n))
    print("missed deadlines 1:", missed_deadlines_stat_1.mean, " with accuracy ", missed_deadlines_stat_1.accuracy())
    print("missed deadlines 2:", missed_deadlines_stat_2.mean, " with accuracy ", missed_deadlines_stat_2.accuracy())
    print("---total missed deadlines:", (missed_deadlines_stat_1.sum() + missed_deadlines_stat_2.sum()) / (
            missed_deadlines_stat_1.n + missed_deadlines_stat_2.n))
    print("scheduler queue length:", scheduler.queue_length_stat.mean, " with accuracy ",
          scheduler.queue_length_stat.accuracy())
    # print("servers queue length:", [s.queue_length_stat for s in scheduler.servers])
    for s in scheduler.servers:
        print("server ", s.id, " queue length:", s.queue_length_stat.mean, " with accuracy ",
              s.queue_length_stat.accuracy())


if __name__ == '__main__':
    main()
