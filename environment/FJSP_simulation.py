import copy
import random

import numpy as np
import pandas as pd
import simpy


class Operation:
    def __init__(self, job_idx: int, op_idx: int, p_k: list):
        self.name = f"Op {job_idx}-{op_idx}"
        self.job_name = f"Job {job_idx}"
        self.processing_time = {}  # dict{available machine : processing time}
        for idx, pt in enumerate(p_k):
            if pt >= 0:
                self.processing_time[idx] = pt


class Job:  # Job = left_op + completed_op 이 통째로 움직임
    def __init__(self, job_idx: int, p_jk: list):
        self.name = f"Job {job_idx}"  # job name, "Job 0"
        self.idx = job_idx
        self.processing_time = p_jk  # processing time list, p[op][machine]
        self.completed_op = []
        self.left_op = []
        self.arrival_time = None  # job arrival time
        self.due_date = 9999999999999
        self.completion_time = 0  # Completion time of lastly completed op => job completion time at the end
        for j in range(len(p_jk)):
            self.left_op.append(Operation(self.idx, j, p_jk[j]))
        # self.past = 0  # 이전 의사결정 시점?
        # self.sink_just = True  # to calculate reward of job in sink
        # self.num_completed_op = 0
        self.sink_just = True  # for calculate decision step reward (tardiness)

    def next_op(self):  # return next op
        return self.left_op[0]

    def op_complete(self):
        self.completed_op.append(self.left_op.pop(0))

    # OP_i : current number of completed operations of Job_i
    def num_completed_ops(self):
        return len(self.completed_op)

    # n_i
    def num_ops(self):
        return len(self.completed_op) + len(self.left_op)

    # CRJ_i = OP_i / n_i
    def completion_rate(self):
        return len(self.completed_op) / (len(self.completed_op) + len(self.left_op))


class Source:  # Generate i-th Job
    def __init__(self, job: Job, env, routing, monitor, p_jk, DDT=1.0, arrival_time=0, num_job_init=0):  # IAT_ave=50
        self.job = job
        self.name = f"Source {job.name}"  # "Source Job i"
        self.env = env
        self.routing = routing
        self.monitor = monitor
        self.p_jk = p_jk
        self.DDT = DDT
        self.arrival_time = arrival_time
        self.num_job_init = num_job_init
        # self.IAT_ave = IAT_ave
        self.non_processed_job = copy.deepcopy(job)
        self.env.process(self.generate())  # iat 없는 경우는 generate에 yield없으므로 self.generate()

    def generate(self):
        yield self.env.timeout(self.arrival_time)
        self.job.arrival_time = self.env.now + self.arrival_time
        # duedate = arrival_time + DDT * expected(avg)_processing_time
        self.job.due_date = self.job.arrival_time + self.DDT * np.sum([np.mean(p_k) for p_k in self.p_jk])
        self.monitor.record(time=self.env.now, jobtype=self.job.name, event="Created", duedate=self.job.due_date)
        # self.job.past = copy.deepcopy(self.env.now)
        self.routing.queue.put(self.job)  # routing 요청
        self.monitor.record(time=self.env.now, jobtype=self.job.name, event="Put in Routing Class", job=self.job.name,
                            queue=[j.idx for j in self.routing.queue.items], duedate=self.job.due_date)
        if self.job.idx + 1 >= self.num_job_init:  # initial job 생성 완료시 process 시작
            if not (str(self.env._queue[0][3]).startswith("<Timeout") and self.env._queue[0][0] == self.env.now):
                yield self.env.process(self.routing.run())


class Process:  # Machine process
    def __init__(self, env, idx: int, sink, routing, monitor):
        self.env = env
        self.idx = idx
        self.name = f"Machine {idx}"  # "Machine j"
        self.sink = sink
        self.routing = routing
        self.monitor = monitor

        self.queue = simpy.Store(env)
        self.idle = True
        self.job_process = None

        self.last_op_completion_time = 0
        self.worked_time_approx = 0

        env.process(self.run())

    # CT_k : completion time of the last op on Machine_k
    def completion_time_last_op(self):
        return self.last_op_completion_time

    # U_k  : utilization rate of Machine_k
    def utilization_rate(self):
        if self.env.now == 0:
            return 0
        return self.worked_time_approx / self.env.now

    def run(self):
        while True:
            self.job_process = yield self.queue.get()
            self.idle = False
            self.monitor.record(time=self.env.now, jobtype=self.job_process.name, event="Work Start",
                                job=self.job_process.next_op().name, machine=self.name,
                                duedate=self.job_process.due_date)
            processing_time = self.job_process.next_op().processing_time[self.idx]
            self.last_op_completion_time = self.env.now + processing_time  # CT_k update
            yield self.env.timeout(processing_time)
            self.worked_time_approx += processing_time
            self.monitor.record(time=self.env.now, jobtype=self.job_process.name, event="Work Finish",
                                job=self.job_process.next_op().name, machine=self.name,
                                duedate=self.job_process.due_date)
            self.job_process.op_complete()  # move just completed op (left_op -> completed_op)
            self.job_process.completion_time = self.env.now

            if len(self.job_process.left_op) == 0:  # Job end -> to sink
                self.sink.put(self.job_process)
            else:  # Job not end -> back to routing queue
                self.routing.queue.put(self.job_process)

            self.idle = True
            self.job_process = None
            if len(self.routing.queue.items) > 0:
                if self.env._queue:  # 동시에 여러 Op이 끝나는 경우 사이에 라우팅을 하지 않도록 함.
                    # if not (str(self.env._queue[0][3]).startswith("<Timeout")):
                    if not (str(self.env._queue[0][3]).startswith("<Timeout") and self.env._queue[0][0] == self.env.now):
                        self.monitor.record(time=self.env.now, event="Request Routing for Job", machine=self.name,
                                            queue=[j.idx for j in self.routing.queue.items])
                        yield self.env.process(self.routing.run())
                else:  # empty simpy queue인 경우는 바로 라우팅
                    self.monitor.record(time=self.env.now, event="Request Routing for Job", machine=self.name,
                                        queue=[j.idx for j in self.routing.queue.items])
                    yield self.env.process(self.routing.run())


class Routing:
    def __init__(self, env, process_dict, monitor):  # source_dict,jt_dict,
        self.env = env
        # self.jt_dict = jt_dict
        self.process_dict = process_dict
        # self.source_dict = source_dict
        self.monitor = monitor

        self.action_mode = None  # self.mapping = {0: "WSPT", 1: "WMDD", 2: "ATC", 3: "WCOVERT"}
        self.mapping = None  # self.action_mode = {'heuristic', ...}
        self.created = 0
        self.queue = simpy.FilterStore(env)  # waiting job queue
        self.indicator = False
        self.decision = None  # decision event indicator

    def run(self):
        # print("Routing-run")
        while True:
            machine_idle = [machine.idle for machine in self.process_dict.values()]
            # print(self.env.now, "#168#", machine_idle)
            if any(machine_idle):  # and len(self.queue.items)
                # print(self.env.now, "#170#", machine_idle)
                self.indicator = True
                self.decision = self.env.event()
                routing_rule = yield self.decision
                self.decision = None
                # self.monitor.record(time=self.env.now, jobtype=job.idx, event="Routing Start", job=job.name,
                #                     memo=f"{routing_rule},  machine {machine_idle.count(True)}개 중 선택")
                # Routing Start, Routing Finish 연속으로 진행돼서 그냥 주석처리함.
                next_job = None
                next_machine = None
                if self.action_mode == 'heuristic':
                    if routing_rule == "rule1":
                        next_job, next_machine = yield self.env.process(self.RULE_1(idle=machine_idle))
                    elif routing_rule == "rule2":
                        next_job, next_machine = yield self.env.process(self.RULE_2(idle=machine_idle))
                    elif routing_rule == "rule3":
                        next_job, next_machine = yield self.env.process(self.RULE_3(idle=machine_idle))
                    elif routing_rule == "rule4":
                        next_job, next_machine = yield self.env.process(self.RULE_4(idle=machine_idle))
                    elif routing_rule == "rule5":
                        next_job, next_machine = yield self.env.process(self.RULE_5(idle=machine_idle))
                    elif routing_rule == "rule6":
                        next_job, next_machine = yield self.env.process(self.RULE_6(idle=machine_idle))
                elif self.action_mode == 'random':
                    next_job, next_machine = yield self.env.process(self.RANDOM(idle=machine_idle))
                elif self.action_mode == 'rule1':
                    next_job, next_machine = yield self.env.process(self.RULE_1(idle=machine_idle))
                elif self.action_mode == 'rule2':
                    next_job, next_machine = yield self.env.process(self.RULE_2(idle=machine_idle))
                elif self.action_mode == 'rule3':
                    next_job, next_machine = yield self.env.process(self.RULE_3(idle=machine_idle))
                elif self.action_mode == 'rule4':
                    next_job, next_machine = yield self.env.process(self.RULE_4(idle=machine_idle))
                elif self.action_mode == 'rule5':
                    next_job, next_machine = yield self.env.process(self.RULE_5(idle=machine_idle))
                elif self.action_mode == 'rule6':
                    next_job, next_machine = yield self.env.process(self.RULE_6(idle=machine_idle))
                elif self.action_mode == 'SPT':
                    next_job, next_machine = yield self.env.process(self.SPT())
                elif self.action_mode == 'LPT':
                    next_job, next_machine = yield self.env.process(self.LPT())
                elif self.action_mode == 'FIFO':
                    next_job, next_machine = yield self.env.process(self.FIFO())
                elif self.action_mode == 'MRT':
                    next_job, next_machine = yield self.env.process(self.MRT())
                elif self.action_mode == 'EDD':
                    next_job, next_machine = yield self.env.process(self.EDD())
                else:
                    exit("no rule-simulation/run")

                if next_job is not None and next_machine is not None:  # Routing 잘 된 경우
                    # print(f"{self.env.now} ", [m.idle for m in self.process_dict.values()])
                    self.monitor.record(time=self.env.now, jobtype=next_job.name, event="Routing Finish",
                                        job=next_job.name, machine=f"Machine {next_machine}",
                                        queue=[j.idx for j in self.queue.items], duedate=next_job.due_date)
                    self.process_dict[f"Machine {next_machine}"].queue.put(next_job)
                    self.process_dict[f"Machine {next_machine}"].idle = False
                    # print(f"{self.env.now} ", [m.idle for m in self.process_dict.values()])
                else:  # Routing 불가z
                    break
            else:  # No Idle Machine
                break

    """
    이하 routing rule(부분 구현 완료)
    RANDOM
    RULE_1
    RULE_2
    RULE_3
    RULE_4
    RULE_5
    RULE_6
    FIFO
    EDD
    MDD
    SPT
    LPT
    """

    def RANDOM(self, idle=None):
        # idle = [machine.idle for machine in self.process_dict.values()]
        # print("idle in rule:", idle)
        idle_idx = [idx for idx, value in enumerate(idle) if value]
        # print("#252#", idle)
        job_list = list(copy.deepcopy(self.queue.items))
        avail_job_list = []
        for job in job_list:
            temp_set = set(job.next_op().processing_time.keys())
            if temp_set.intersection(set(idle_idx)):
                avail_job_list.append(job)
        if avail_job_list is not None and len(avail_job_list) > 0:
            next_job_name = random.choice(avail_job_list).name
            next_job = yield self.queue.get(lambda x: x.name == next_job_name)
            temp_set = set(next_job.next_op().processing_time.keys())
            next_machine = random.choice(list(temp_set.intersection(set(idle_idx))))
            return next_job, next_machine
        return None, None

    def RULE_1(self, idle=None):
        idle = [machine.idle for machine in self.process_dict.values()]
        idle_idx = [idx for idx, value in enumerate(idle) if value]
        job_list = list(copy.deepcopy(self.queue.items))
        avail_job_list = []
        for job in job_list:
            temp_set = set(job.next_op().processing_time.keys())
            if temp_set.intersection(set(idle_idx)):
                avail_job_list.append(job)
        if avail_job_list is not None and len(avail_job_list) > 0:
            T_cur = np.mean([m.completion_time_last_op() for m in self.process_dict.values()])
            Tard_job = []
            # UC_job = []
            for j in avail_job_list:
                if j.num_completed_ops() < j.num_ops() and j.due_date < T_cur:
                    Tard_job.append(j)

            UC_job = avail_job_list
            if not Tard_job:  # isempty Tard_job
                next_job = min(UC_job, key=lambda j: (j.due_date - T_cur) / (j.num_ops() - j.num_completed_ops()))
            else:
                next_job = max(Tard_job, key=lambda j: T_cur + np.sum(
                    [np.mean(list(left_op.processing_time.values())) for left_op in j.left_op]) - j.due_date)
            next_job_name = next_job.name
            next_job = yield self.queue.get(lambda x: x.name == next_job_name)
            temp_set = set(next_job.next_op().processing_time.keys())
            # next_machine = random.choice(list(temp_set.intersection(set(idle_idx))))
            next_machine = min(list(temp_set.intersection(set(idle_idx))),
                               key=lambda m: max(self.process_dict[f"Machine {m}"].completion_time_last_op(),
                                                 next_job.completion_time, next_job.arrival_time))

        else:
            return None, None
        return next_job, next_machine

    def RULE_2(self, idle=None):
        idle = [machine.idle for machine in self.process_dict.values()]
        idle_idx = [idx for idx, value in enumerate(idle) if value]
        job_list = list(copy.deepcopy(self.queue.items))
        avail_job_list = []
        for job in job_list:
            temp_set = set(job.next_op().processing_time.keys())
            if temp_set.intersection(set(idle_idx)):
                avail_job_list.append(job)
        if avail_job_list is not None and len(avail_job_list) > 0:
            T_cur = np.mean([m.completion_time_last_op() for m in self.process_dict.values()])
            Tard_job = []
            for j in avail_job_list:
                if j.num_completed_ops() < j.num_ops() and j.due_date < T_cur:
                    Tard_job.append(j)
            UC_job = avail_job_list
            if not Tard_job:  # isempty Tard_job
                next_job = min(UC_job, key=lambda j: (j.due_date - T_cur) / np.sum(
                    [np.mean(list(left_op.processing_time.values())) for left_op in j.left_op]))
            else:
                next_job = max(Tard_job, key=lambda j: T_cur + np.sum(
                    [np.mean(list(left_op.processing_time.values())) for left_op in j.left_op]) - j.due_date)
            next_job_name = next_job.name
            next_job = yield self.queue.get(lambda x: x.name == next_job_name)
            temp_set = set(next_job.next_op().processing_time.keys())
            # next_machine = random.choice(list(temp_set.intersection(set(idle_idx))))
            next_machine = min(list(temp_set.intersection(set(idle_idx))),
                               key=lambda m: max(self.process_dict[f"Machine {m}"].completion_time_last_op(),
                                                 next_job.completion_time, next_job.arrival_time))
        else:
            return None, None
        return next_job, next_machine

    def RULE_3(self, idle=None):
        idle = [machine.idle for machine in self.process_dict.values()]
        idle_idx = [idx for idx, value in enumerate(idle) if value]
        job_list = list(copy.deepcopy(self.queue.items))
        avail_job_list = []
        for job in job_list:
            temp_set = set(job.next_op().processing_time.keys())
            if temp_set.intersection(set(idle_idx)):
                avail_job_list.append(job)
        if avail_job_list is not None and len(avail_job_list) > 0:
            T_cur = np.mean([m.completion_time_last_op() for m in self.process_dict.values()])
            UC_job = avail_job_list
            next_job = max(UC_job, key=lambda j: T_cur + np.sum(
                [np.mean(list(left_op.processing_time.values())) for left_op in j.left_op]) - j.due_date)
            next_job_name = next_job.name
            next_job = yield self.queue.get(lambda x: x.name == next_job_name)

            temp_set = set(next_job.next_op().processing_time.keys())
            r = random.uniform(0, 1.0)
            if r < 0.5:
                next_machine = min(list(temp_set.intersection(set(idle_idx))),
                                   key=lambda m: self.process_dict[f"Machine {m}"].worked_time_approx /
                                                 self.process_dict[f"Machine {m}"].completion_time_last_op() if
                                   self.process_dict[f"Machine {m}"].completion_time_last_op() != 0 else 99999999999999)
            else:
                next_machine = min(list(temp_set.intersection(set(idle_idx))),
                                   key=lambda m: self.process_dict[f"Machine {m}"].worked_time_approx)
        else:
            return None, None
        return next_job, next_machine

    def RULE_4(self, idle=None):
        idle = [machine.idle for machine in self.process_dict.values()]
        idle_idx = [idx for idx, value in enumerate(idle) if value]
        job_list = list(copy.deepcopy(self.queue.items))
        avail_job_list = []
        for job in job_list:
            temp_set = set(job.next_op().processing_time.keys())
            if temp_set.intersection(set(idle_idx)):
                avail_job_list.append(job)
        if avail_job_list is not None and len(avail_job_list) > 0:
            T_cur = np.mean([m.completion_time_last_op() for m in self.process_dict.values()])
            Tard_job = []
            for j in avail_job_list:
                if j.num_completed_ops() < j.num_ops() and j.due_date < T_cur:
                    Tard_job.append(j)
            # UC_job = avail_job_list
            next_job_name = random.choice(avail_job_list).name
            next_job = yield self.queue.get(lambda x: x.name == next_job_name)
            temp_set = set(next_job.next_op().processing_time.keys())
            # next_machine = random.choice(list(temp_set.intersection(set(idle_idx))))

            next_machine = min(list(temp_set.intersection(set(idle_idx))),
                               key=lambda m: max(self.process_dict[f"Machine {m}"].completion_time_last_op(),
                                                 next_job.completion_time, next_job.arrival_time))

        else:
            return None, None
        return next_job, next_machine

    def RULE_5(self, idle=None):
        idle = [machine.idle for machine in self.process_dict.values()]
        idle_idx = [idx for idx, value in enumerate(idle) if value]
        job_list = list(copy.deepcopy(self.queue.items))
        avail_job_list = []
        for job in job_list:
            temp_set = set(job.next_op().processing_time.keys())
            if temp_set.intersection(set(idle_idx)):
                avail_job_list.append(job)
        if avail_job_list is not None and len(avail_job_list) > 0:
            T_cur = np.mean([m.completion_time_last_op() for m in self.process_dict.values()])
            Tard_job = []
            for j in avail_job_list:
                if j.num_completed_ops() < j.num_ops() and j.due_date < T_cur:
                    Tard_job.append(j)
            # UC_job = avail_job_list
            UC_job = avail_job_list
            if not Tard_job:  # isempty Tard_job
                next_job = min(UC_job, key=lambda j: j.num_completed_ops() / j.num_ops() * (j.due_date - T_cur))
            else:
                # 뭔가 규칙이 이상한디......
                next_job = max(Tard_job, key=lambda j: j.num_ops() / j.num_completed_ops() * (T_cur + np.sum(
                    [np.mean(list(left_op.processing_time.values())) for left_op in
                     j.left_op]) - j.due_date) if j.num_completed_ops() != 0 else 99999999999999)
            next_job_name = next_job.name
            next_job = yield self.queue.get(lambda x: x.name == next_job_name)
            temp_set = set(next_job.next_op().processing_time.keys())
            next_machine = random.choice(list(temp_set.intersection(set(idle_idx))))
        else:
            return None, None
        return next_job, next_machine

    def RULE_6(self, idle=None):
        idle = [machine.idle for machine in self.process_dict.values()]
        idle_idx = [idx for idx, value in enumerate(idle) if value]
        job_list = list(copy.deepcopy(self.queue.items))
        avail_job_list = []
        for job in job_list:
            temp_set = set(job.next_op().processing_time.keys())
            if temp_set.intersection(set(idle_idx)):
                avail_job_list.append(job)
        if avail_job_list is not None and len(avail_job_list) > 0:
            T_cur = np.mean([m.completion_time_last_op() for m in self.process_dict.values()])
            # UC_job = avail_job_list
            UC_job = avail_job_list
            next_job = max(UC_job, key=lambda j: T_cur + np.sum(
                [np.mean(list(left_op.processing_time.values())) for left_op in j.left_op]) - j.due_date)
            next_job_name = next_job.name
            next_job = yield self.queue.get(lambda x: x.name == next_job_name)
            temp_set = set(next_job.next_op().processing_time.keys())
            next_machine = random.choice(list(temp_set.intersection(set(idle_idx))))
        else:
            return None, None
        return next_job, next_machine

    def FIFO(self):
        idle = [machine.idle for machine in self.process_dict.values()]
        idle_idx = [idx for idx, value in enumerate(idle) if value]
        job_list = list(copy.deepcopy(self.queue.items))

        next_job_name = None
        next_machine = None
        for job in job_list:
            print("FIFO job_list job.arrivaltime : ", job.arrivaltime)
            temp_set = set(job.next_op().processing_time.keys())
            next_machine = random.choice(list(temp_set.intersection(set(idle_idx))))
            if temp_set.intersection(set(idle_idx)):
                next_job_name = job.name

        next_job = yield self.queue.get(lambda x: x.name == next_job_name)
        next_machine = min(list(temp_set.intersection(set(idle_idx))),
                           key=lambda m: max(self.process_dict[f"Machine {m}"].completion_time_last_op(),
                                             next_job.completion_time, next_job.arrival_time))
        return next_job, next_machine

    def EDD(self):
        idle = [machine.idle for machine in self.process_dict.values()]
        idle_idx = [idx for idx, value in enumerate(idle) if value]
        job_list = list(copy.deepcopy(self.queue.items))
        avail_job_list = []
        for job in job_list:
            temp_set = set(job.next_op().processing_time.keys())
            if temp_set.intersection(set(idle_idx)):
                avail_job_list.append(job)
        next_job = None
        if avail_job_list is not None and len(avail_job_list) > 0:
            dd = 9999999999999999
            for j in avail_job_list:
                if dd > j.due_date:
                    dd = j.due_date
                    next_job = j

            next_job_name = next_job.name
            next_job = yield self.queue.get(lambda x: x.name == next_job_name)
            temp_set = set(next_job.next_op().processing_time.keys())
            next_machine = min(list(temp_set.intersection(set(idle_idx))),
                               key=lambda m: max(self.process_dict[f"Machine {m}"].completion_time_last_op(),
                                                 next_job.completion_time, next_job.arrival_time))

        else:
            return None, None
        return next_job, next_machine

    def MRT(self):
        idle = [machine.idle for machine in self.process_dict.values()]
        idle_idx = [idx for idx, value in enumerate(idle) if value]
        job_list = list(copy.deepcopy(self.queue.items))
        avail_job_list = []
        for job in job_list:
            temp_set = set(job.next_op().processing_time.keys())
            if temp_set.intersection(set(idle_idx)):
                avail_job_list.append(job)
        if avail_job_list is not None and len(avail_job_list) > 0:
            RT = 0
            for j in avail_job_list:
                rt = np.sum([np.mean(list(op.processing_time.values())) for op in j.left_op])
                if RT < rt:
                    RT = rt
                    next_job = j

            next_job_name = next_job.name
            next_job = yield self.queue.get(lambda x: x.name == next_job_name)
            temp_set = set(next_job.next_op().processing_time.keys())

            next_machine = min(list(temp_set.intersection(set(idle_idx))),
                               key=lambda m: next_job.next_op().processing_time[m])

            # next_machine = min(list(temp_set.intersection(set(idle_idx))),
            #                    key=lambda m: max(self.process_dict[f"Machine {m}"].completion_time_last_op(),
            #                                      next_job.completion_time, next_job.arrival_time))
        else:
            return None, None
        return next_job, next_machine

    def SPT(self):
        idle = [machine.idle for machine in self.process_dict.values()]
        idle_idx = [idx for idx, value in enumerate(idle) if value]
        job_list = list(copy.deepcopy(self.queue.items))
        avail_job_list = []
        for job in job_list:
            temp_set = set(job.next_op().processing_time.keys())
            if temp_set.intersection(set(idle_idx)):
                avail_job_list.append(job)
        if avail_job_list is not None and len(avail_job_list) > 0:
            min_pt = 9999999999
            for j in avail_job_list:
                for i, p in j.next_op().processing_time.items():
                    if p < min_pt:
                        min_pt = p
                        next_job = j
                        next_machine = i

            next_job_name = next_job.name
            next_job = yield self.queue.get(lambda x: x.name == next_job_name)
        else:
            return None, None
        return next_job, next_machine

    def LPT(self):
        idle = [machine.idle for machine in self.process_dict.values()]
        idle_idx = [idx for idx, value in enumerate(idle) if value]
        job_list = list(copy.deepcopy(self.queue.items))
        avail_job_list = []
        for job in job_list:
            temp_set = set(job.next_op().processing_time.keys())
            if temp_set.intersection(set(idle_idx)):
                avail_job_list.append(job)
        if avail_job_list is not None and len(avail_job_list) > 0:
            max_pt = 0
            for j in avail_job_list:
                for i, p in j.next_op().processing_time.items():
                    if p > max_pt:
                        max_pt = p
                        next_job = j
                        next_machine = i

            next_job_name = next_job.name
            next_job = yield self.queue.get(lambda x: x.name == next_job_name)

        else:
            return None, None
        return next_job, next_machine


class Sink:
    def __init__(self, env, monitor, source_dict, num_job):  # end_num, jt_dict,
        self.env = env
        self.monitor = monitor
        # self.jt_dict = jt_dict
        self.source_dict = source_dict
        self.num_left_job = num_job
        self.job_list = []

    def put(self, job):
        self.num_left_job -= 1
        self.monitor.record(time=self.env.now, jobtype=job.name, event="Completed", job=job.name,
                            tard=max(0, self.env.now - job.due_date), duedate=job.due_date)
        self.monitor.tardiness += min(0, job.due_date - self.env.now)
        self.job_list.append(job)
        # print(self.monitor.tardiness)


class Monitor:
    def __init__(self, filepath):
        self.time = list()
        self.jobtype = list()
        self.event = list()
        self.job = list()
        self.machine = list()
        self.queue = list()
        self.memo = list()
        self.weight = list()
        self.tard = list()
        self.duedate = list()

        self.filepath = filepath

        self.tardiness = None

    def record(self, time=None, jobtype=None, event=None, job=None, machine=None, queue=None, memo=None, tard=None,
               duedate=None):
        self.time.append(round(time, 2))
        self.jobtype.append(jobtype)
        self.event.append(event)
        self.job.append(job)
        self.machine.append(machine)
        self.queue.append(queue)
        self.memo.append(memo)
        # self.weight.append(weight)
        self.tard.append(tard)
        self.duedate.append(duedate)

    def save_tracer(self):
        event_tracer = pd.DataFrame(
            columns=["Time", "JobType", "Event", "Job", "Machine", "Queue", "Memo", "Tardiness", "DueDate"])
        event_tracer["Time"] = self.time
        event_tracer["JobType"] = self.jobtype
        event_tracer["Event"] = self.event
        event_tracer["Job"] = self.job
        event_tracer["Machine"] = self.machine
        event_tracer["Queue"] = self.queue
        event_tracer["Memo"] = self.memo
        # event_tracer["weight"] = self.weight
        event_tracer["Tardiness"] = self.tard
        event_tracer["DueDate"] = self.duedate
        event_tracer.to_csv(r'{}'.format(self.filepath))

    def reset(self):
        self.time = list()
        self.jobtype = list()
        self.event = list()
        self.job = list()
        self.machine = list()
        self.queue = list()
        self.memo = list()
        self.weight = []
        self.tard = []
        self.duedate = []

        self.tardiness = 0
