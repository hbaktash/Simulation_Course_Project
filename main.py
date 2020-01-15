import os
import math
import numpy as np
from typing import List

import utils

PATH = "C:\\Users\\hosse\\Desktop\\Simulation_Course_Project"
# PATH = "C:\\Users\\Amin\\Desktop\\Simulation_Course_Project-master"
TASK_COUNT = 5000000  # number of tasks to enter system
TYPE_PARAM = 0.1  # probabilty of task type 1
MAX_TASKS = 50000000  # tasks limit
IGNORED_TASKS = 1000  # number of Unstable phase tasks
DELTA_T = 0.45  # Global time Delta

time = 0
out_tasks = 0  # number of tasks that left system (done/missed)
update_stats = False  # Unstable/stable phase flag


class Data:  # Statistics class
    def __init__(self, name: str):
        self.name = name
        self.n: int = 0
        self.mean: float = 0
        self.std: float = 0
        self.last_data = None

    def update(self, new_data: float):  # update mean/std/n of associated data
        global update_stats
        if update_stats:
            self.mean = ((self.mean * self.n) + new_data) / (self.n + 1)
            if self.n > 2:
                self.std = math.sqrt(((self.std ** 2) * (self.n - 1) + (new_data - self.mean) ** 2) / self.n)
            elif self.n == 2:
                self.std = math.sqrt((new_data - self.mean) ** 2 + (self.last_data - self.mean) ** 2)
            self.n += 1
            self.last_data = new_data

    def accuracy(self):  # returns accuracy of data
        if self.n >= 2:
            if self.mean == 0:  # Zero division
                return 0
            return 1.96 * self.std / (math.sqrt(self.n) * self.mean)
        return 1

    def sum(self):  # sum of avg
        return self.n * self.mean


system_time_stat_1 = Data("sys time 1")  # Type 1 task system time
system_time_stat_2 = Data("sys time 2")  # Type 2 task system time
queue_time_stat_1 = Data("q time 1")  # Type 1 task queue time
queue_time_stat_2 = Data("q time 2")  # Type 2 task queue time
missed_deadlines_stat_1 = Data("missed deadlines 1")  # Type 1 task missed deadlines
missed_deadlines_stat_2 = Data("missed deadlines 2")  # Type 2 task missed deadlines


class Task:  # Tasks class including stats of an individual task
    def __init__(self, task_type, deadline, arrival_time, id=-1):
        self.id = id
        self.arrival_time = arrival_time
        self.server_process_time = -1
        self.scheduler_process_time = -1
        self.deadline = deadline
        self.q1_time = -1  # scheduler queue
        self.q2_time = -1  # server queue
        self.type = task_type
        self.core = None

    def print_task_data(self):  # returns stats - Debugging
        print(
            "task {0}\n    arrival time: {1}\n    deadline: {2}\n    scheduler process: {3}\n    server process:{4}\n    q1: {5}\n    q2: {6}".format(
                self.id, self.arrival_time, self.deadline, self.scheduler_process_time, self.server_process_time,
                self.q1_time,
                self.q2_time))


class Core:  # Cores class including rate, idle state and current task
    def __init__(self, mu):
        self.idle = True
        self.mu = mu
        self.current_task: Task = None
        self.current_start_time = -1
        self.current_exec_time = -1  # generated with mu

    def insert_new_task(self, task: Task):  # Accept new task and generate rate
        if not self.idle:
            print("Nani?")
        else:
            self.idle = False
            self.current_task = task
            self.current_start_time = time
            self.current_exec_time = round(utils.generate_exponential(self.mu))
            self.current_task.server_process_time = self.current_exec_time
            self.current_task.core = self

    def update(self):  # update statistics if task is finished
        global out_tasks, system_time_stat_1, system_time_stat_2, missed_deadlines_stat_1, missed_deadlines_stat_2
        if self.idle:
            return
        if time >= self.current_exec_time + self.current_start_time:  # if task is finished
            if self.current_task.type == 1:  # update type 1 stats
                system_time_stat_1.update(self.current_task.q1_time +
                                          self.current_task.q2_time +
                                          self.current_task.server_process_time +
                                          self.current_task.scheduler_process_time)
                missed_deadlines_stat_1.update(0)
            else:  # update type 2 stats
                system_time_stat_2.update(self.current_task.q1_time +
                                          self.current_task.q2_time +
                                          self.current_task.server_process_time +
                                          self.current_task.scheduler_process_time)
                missed_deadlines_stat_2.update(0)
            self.current_task = None
            self.idle = True
            self.current_exec_time = -1
            self.current_start_time = -1
            out_tasks += 1  # 1 task is done


class Server:  # servers class including it's cores/queue and checking deadlines
    def __init__(self, cores: List[Core], id: int):
        self.id = id
        self.cores = cores
        self.queue: List[Task] = []
        self.queue_length_stat = Data("server {} queue len".format(self.id))  # server queue length statistics

    def handle_queue(self):  # checkout tasks in queue and assign to a core if it's idle
        global out_tasks, queue_time_stat_1, queue_time_stat_2, missed_deadlines_stat_1, missed_deadlines_stat_2, system_time_stat_1, system_time_stat_2
        inserted_count = 0
        for task in self.queue:
            if time - task.arrival_time > task.deadline:  # if missed deadline
                inserted_count += 1
                out_tasks += 1
                task.q2_time = task.deadline - task.q1_time - task.scheduler_process_time  # update task server queue time
                if task.type == 1:  # update type 1 stats
                    queue_time_stat_1.update(task.q1_time + task.q2_time)
                    missed_deadlines_stat_1.update(1)
                    system_time_stat_1.update(task.q1_time + task.q2_time + task.scheduler_process_time)
                else:  # update type 2 stats
                    queue_time_stat_2.update(task.q1_time + task.q2_time)
                    missed_deadlines_stat_2.update(1)
                    system_time_stat_2.update(task.q1_time + task.q2_time + task.scheduler_process_time)
                continue
            for core in self.cores:  # search cores for an idle core
                if core.idle:  # assign task to idle core and update stats
                    task.q2_time = time - task.arrival_time - task.q1_time - task.scheduler_process_time
                    core.insert_new_task(task)
                    inserted_count += 1
                    if task.type == 1:  # update type 1 statistics
                        queue_time_stat_1.update(task.q1_time + task.q2_time)
                    else:  # update type 2 statistics
                        queue_time_stat_2.update(task.q1_time + task.q2_time)
                    # print(task.id)
                    break
            if task.core is None:  # if there is no idle core, no need to check other tasks in queue
                break
        self.queue = self.queue[inserted_count:]  # remove assigned tasks from queue
        self.queue_length_stat.update(len(self.queue))


class Scheduler:  # Scheduler class with 2 queue for type 1-2 tasks / have access to all servers/cores
    def __init__(self, mu, servers: List[Server]):
        self.servers: List[Server] = servers
        self.rate = mu
        self.idle = True
        self.current_task: Task = None  # task in scheduler core
        self.next_free_time = 0  # related to rate
        self.queue1: List[Task] = []  # high priority queue
        self.queue2: List[Task] = []  # low priority queue
        self.queue_length_stat = Data("scheduler queue len")  # scheduler queue length statistics

    def handle_queue(self):  # number of tasks to send in and send them
        global out_tasks, queue_time_stat_1, queue_time_stat_2, missed_deadlines_stat_1, missed_deadlines_stat_2, system_time_stat_1, system_time_stat_2
        if time >= self.next_free_time:  # if last task is scheduled (rate dependent)
            if len(self.queue1) != 0:  # take a task from type 1 queue and update queue
                current_task: Task = self.queue1[0]
                self.queue1 = self.queue1[1:]
            elif len(self.queue2) != 0:  # with less priority take a task from type 2 queue and update queue
                current_task: Task = self.queue2[0]
                self.queue2 = self.queue2[1:]
            else:
                return
            service_time = (utils.generate_exponential(self.rate))  # time needed to schedule task
            # print("Service time:", service_time)
            self.next_free_time = time + service_time  # scheduler is free after task service time
            if time - current_task.arrival_time > current_task.deadline:  # check deadline
                out_tasks += 1
                current_task.q1_time = time - current_task.arrival_time  # waited in queue more than deadline    !!!current_task.deadline!!!
                if current_task.type == 1:  # update type 1 statistics
                    queue_time_stat_1.update(current_task.q1_time)
                    missed_deadlines_stat_1.update(1)
                    system_time_stat_1.update(current_task.q1_time)
                else:  # update type 2 statistics
                    queue_time_stat_2.update(current_task.q1_time)
                    missed_deadlines_stat_2.update(1)
                    system_time_stat_2.update(current_task.q1_time)
                self.handle_queue()  # skip the missed deadline task
            elif time - current_task.arrival_time + service_time > current_task.deadline:  # if deadline will pass mid scheduling
                out_tasks += 1
                current_task.q1_time = time - current_task.arrival_time
                current_task.scheduler_process_time = current_task.deadline - current_task.q1_time  # it's proccessed until deadline
                if current_task.type == 1:  # update type 1 statistics
                    queue_time_stat_1.update(current_task.q1_time)
                    missed_deadlines_stat_1.update(1)
                    system_time_stat_1.update(current_task.deadline)
                else:  # update type 2 statistics
                    queue_time_stat_2.update(current_task.q1_time)
                    missed_deadlines_stat_2.update(1)
                    system_time_stat_2.update(current_task.deadline)
                self.next_free_time = time + current_task.deadline  # next free time is set to deadline of task
                self.current_task = None
            else:  # if deadline is hit (not missed)
                current_task.q1_time = time - current_task.arrival_time
                current_task.scheduler_process_time = service_time
                self.current_task = current_task  # set task in queue to scheduler task
                self.queue_length_stat.update(len(self.queue1) + len(self.queue2))  # update statistics

    def assign_task(self):  # assign task to a server
        self.servers.sort(key=lambda x: len(x.queue))  # sort servers from least queue length
        min_length = len(self.servers[0].queue)
        min_server_lenghts = []
        for server in self.servers:  # search servers for same queue length as min
            if len(server.queue) == min_length:
                min_server_lenghts.append(server)
            else:
                break
        rand_num = np.random.randint(0, len(min_server_lenghts))  # pick a random server with least queue length
        # print("server id:", self.servers[rand_num].id)
        self.servers[rand_num].queue.append(self.current_task)  # assign task to server

    def update(self, tasks_to_insert: List[Task]):  # handling assign_task / handle_queue / new arrivals
        if self.current_task is None:
            pass
        else:
            if time >= self.next_free_time:  # if scheduling last task is finished
                self.assign_task()
                self.current_task = None  # remove assigned task from scheduler core

        self.insert_new_tasks(tasks_to_insert)  # insert new task to queue (new arrivals)
        self.handle_queue()

        if self.current_task is None:
            pass
        else:  # have to check again after inserting new tasks
            if time >= self.next_free_time:
                self.assign_task()
                self.current_task = None

    def update_servers(
            self):  # calling servers handle_queue (checkout tasks in queue and assign to a core if it's idle)
        for server in self.servers:
            server.handle_queue()

    def update_cores(self):  # calling cores update (update statistics if task is finished)
        for server in self.servers:
            for core in server.cores:
                core.update()

    def insert_new_tasks(self, tasks_to_insert):  # insert new arrivals to queue 1&2
        for task in tasks_to_insert:
            if task.type == 1:
                self.queue1.append(task)
            else:
                self.queue2.append(task)

    def all_accurate(self):  # Check 0.05 accuracy for all statistics
        global queue_time_stat_1, queue_time_stat_2, missed_deadlines_stat_1, missed_deadlines_stat_2, system_time_stat_1, system_time_stat_2
        # count = 7 + len(self.servers)
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


def initialize_servers(file_name: str):  # read from file and create servers
    servers = []
    with open(os.path.join(PATH, file_name)) as f:
        lines = f.readlines()
        first_params = list(map(float, lines[0].split(" ")))
        arrival_lamda = first_params[1]  # lamda
        deadline_alpha = first_params[2]  # alpha
        scheduler_mu = first_params[3]  # mu
        j = 0
        for line in lines[1:]:  # for M (servers)
            params = list(map(float, line.split(" ")))
            cores = [Core(mu) for mu in params[1:]]  # create list of N cores
            servers.append(Server(cores, j))  # create server with N cores and cores ID
            j += 1
    scheduler = Scheduler(scheduler_mu, servers)  # create scheduler with given mu and access to all servers
    return scheduler, arrival_lamda, deadline_alpha


def task_generator(arrival_lamda, deadline_alpha, task_count=TASK_COUNT):  # generate all tasks
    tasks: List[Task] = []
    t = 0
    for i in range(task_count):
        type = utils.generate_type(TYPE_PARAM)  # generate type with probability 0.1 for 1
        deadline = (utils.generate_exponential(1 / deadline_alpha))  # generate deadline for task i
        inter_arrival = (utils.generate_exponential(1 / arrival_lamda))  # generate inter-arrival for task i
        arrival = t + inter_arrival  # definite arrival time for task i
        temp_task = Task(type, deadline, arrival, id=i)  # create task i
        tasks.append(temp_task)  # these tasks are sorted by arrival
        t += inter_arrival  # time (not the global time)
    return tasks


def insertable_tasks(tasks: List[Task], current_time, current_task_pointer):  # find tasks that arrived
    tasks_to_add = []
    i = current_task_pointer
    while i < len(tasks):  # search tasks that we didn't search before
        if tasks[i].arrival_time <= current_time:  # if it's arrived
            tasks_to_add.append(tasks[i])
            i += 1
        elif tasks[i].arrival_time > current_time:
            break
    return tasks_to_add


def start(scheduler: Scheduler, tasks: List[Task]):  # system While
    global time, update_stats, out_tasks
    current_task_pointer = 0
    while scheduler.all_accurate() or (out_tasks < MAX_TASKS and out_tasks < len(
            tasks)):  # repeat until 0.05 accuracy or limit (or until all finished)
        scheduler.update_cores()  # update statistics if task is finished
        scheduler.update_servers()  # servers handle_queue
        tasks_to_insert = insertable_tasks(tasks, time, current_task_pointer)  # check arrivals
        current_task_pointer += len(tasks_to_insert)
        scheduler.update(tasks_to_insert)  # handling assign_task / scheduler handle_queue / new arrivals
        scheduler.update_servers()  # update again after new arrivals
        scheduler.update_cores()  # update again after new arrivals
        if out_tasks % 1000 == 0:
            print("outs: ", out_tasks)  # number of tasks done
        if out_tasks >= IGNORED_TASKS:  # ignore first X tasks
            update_stats = True
        time += DELTA_T  # global time


def print_info(tasks: List[Task]):  # Debugging
    i = 0
    for task in tasks:
        task.print_task_data()
        i += 1


def main():
    global queue_time_stat_1, queue_time_stat_2, missed_deadlines_stat_1, missed_deadlines_stat_2, system_time_stat_1, system_time_stat_2
    scheduler, arrival_lamda, deadline_alpha = initialize_servers("input.txt")
    tasks = task_generator(arrival_lamda, deadline_alpha, task_count=TASK_COUNT)
    start(scheduler, tasks)  # calls system while and starts system
    # print_info(tasks)
    print("\n\n\n###Statistics###")  ######### Prints Statistics #########
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
