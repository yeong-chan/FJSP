import re
from typing import Tuple

from simpy.core import Environment

from environment.FJSP_simulation import *

"""
PARAMETER SETTINGS in Paper "Dynamic scheduling for flexible job shop with new job insertions by DRL(2020)"
https://www.sciencedirect.com/science/article/pii/S1568494620301484

Number of machines, m = {10, 20, 30, 40, 50}
Number of available machines of each operation: U(0, M]
Number of initial jobs at beginning, n_ini = 20
Total number of new inserted jobs, n_add = {50, 100, 200}
Due date tightness, DDT = {0.5, 1.0, 1.5}
Number of operations belonging to a job: U(0, 20]
Processing time of an operation on an available machine, t_ijk: U(0,50]
IAT(poisson dist, avg value), IAT_ave = {50, 100, 200}
"""


class FJSP:
    def __init__(self, num_m=10, num_job_init=3, num_job_add=20, DDT=1.0, IAT_ave=50, action_mode='random',
                 log_dir=None):
        self.num_machine = num_m
        self.num_job_init = num_job_init
        self.num_job_add = num_job_add
        self.num_job = num_job_init + num_job_add
        self.DDT = DDT
        self.IAT_ave = IAT_ave
        self.log_dir = log_dir

        self.p_ijk, self.p_ij = self._generating_data()
        self.e = 0  # for log idx
        self.time = 0
        self.last_decision_time = 0  # last decision time
        self.action_mode = action_mode
        if self.action_mode == "heuristic":
            # self.mapping = {0: "rule1", 1: "rule2", 2: "rule3", 3: "rule4", 4: "rule5", 5: "rule6"}
            self.mapping = {0: "rule1", 1: "rule2", 2: "rule3", 3: "rule4", 4: "rule5", 5: "rule6", 6:"SPT", 7:"LPT"}
        if self.action_mode == "random":
            self.mapping = {0: "RANDOM"}
        elif self.action_mode == "rule1":
            self.mapping = {0: "rule1"}
        elif self.action_mode == "rule2":
            self.mapping = {0: "rule2"}
        elif self.action_mode == "rule3":
            self.mapping = {0: "rule3"}
        elif self.action_mode == "rule4":
            self.mapping = {0: "rule4"}
        elif self.action_mode == "rule5":
            self.mapping = {0: "rule5"}
        elif self.action_mode == "rule6":
            self.mapping = {0: "rule6"}
        elif self.action_mode == "SPT":
            self.mapping = {0: "SPT"}
        elif self.action_mode == "LPT":
            self.mapping = {0: "LPT"}
        elif self.action_mode == "EDD":
            self.mapping = {0: "EDD"}
        elif self.action_mode == "FIFO":
            self.mapping = {0: "FIFO"}
        elif self.action_mode == "MRT":
            self.mapping = {0: "MRT"}

        # _modeling
        self.sim_env, self.process_dict, self.source_dict, self.jt_dict, self.sink, self.routing, self.monitor = self._modeling()

    def step(self, action: int = 0) -> Tuple[np.ndarray, float, bool]:
        done = False

        routing_rule = self.mapping[action]
        #print(self.mapping[0], routing_rule)
        self.routing.decision.succeed(routing_rule)
        self.routing.indicator = False
        while True:
            if self.routing.indicator:  # Routing 해야할 때
                if self.sim_env.now != self.time:
                    self.time = self.sim_env.now  # decision time update
                break
            if self.sink.num_left_job == 0:  # episode end
                done = True

                self.sim_env.run()
                self.monitor.save_tracer()
                break
            if len(self.sim_env._queue) == 0:  # Empty simpy queue
                self.monitor.save_tracer()
                exit("empty queue error")
            self.sim_env.step()
        reward = self._calculate_reward()
        next_state = self._get_state()
        return next_state, reward, done

    def reset(self) -> np.ndarray:
        self.p_ijk, self.p_ij = self._generating_data()
        self.e = self.e + 1 if self.e > 0 else 1  # episode
        self.time = 0
        self.last_decision_time = 0

        self.sim_env, self.process_dict, self.source_dict, self.jt_dict, self.sink, self.routing, self.monitor = self._modeling()
        self.monitor.reset()

        while True:
            # Check whether there is any decision time step
            if self.routing.indicator:
                break
            self.sim_env.step()

        return self._get_state()

    def _modeling(self) -> Tuple[Environment, dict, dict, dict, Sink, Routing, Monitor]:
        env = simpy.Environment()
        process_dict = dict()  # {"Machine 0": Process class, ...}
        source_dict = dict()  # {"Source 0": Source class, ...}
        jt_dict = dict()  # {"Job 0" : [Job class, ... ], ... }
        for i in range(self.num_job):
            for j in range(len(self.p_ijk[i])):
                jt_dict[f"Job {i}"] = Job(i, self.p_ijk[i])
        monitor = Monitor(self.log_dir + '\\log_%d.csv ' % self.e)
        routing = Routing(env, process_dict, monitor)  # source_dict, jt_dict,
        routing.action_mode = self.action_mode
        routing.mapping = self.mapping

        iat = np.random.exponential(self.IAT_ave, self.num_job_add)
        iat = np.ceil(iat).astype(int)  # iat 올림
        for jt_name, job in jt_dict.items():
            # idx = int(jt_name[-2:]) if jt_name[-2].isdigit() else int(jt_name[-1])  # job idx
            idx = int(re.findall(r'\d+', jt_name)[-1])
            if idx < self.num_job_init:
                arrival_time = 0
            else:
                arrival_time = iat[idx - self.num_job_init]
            source_dict[f"Source {idx}"] = Source(jt_dict[f"Job {idx}"], env, routing, monitor, self.p_ijk[idx],
                                                  self.DDT, arrival_time, self.num_job_init)

        # initial job 생성 완료시 source class에서 env.process(self.routing.run()) 실행
        # initial job 없는 문제는 Source 다 만들고 나서 routing env.process(routing.run())

        sink = Sink(env, monitor, source_dict, self.num_job)  # end_num = self.num_job, jt_dict,

        for j in range(self.num_machine):
            process_dict[f"Machine {j}"] = Process(env, j, sink, routing, monitor)

        return env, process_dict, source_dict, jt_dict, sink, routing, monitor

    def _calculate_reward(self) -> float:
        reward = 0
        now = copy.deepcopy(self.sim_env.now)
        # real tardiness during single decision step
        if now != self.last_decision_time:
            # print(f"{now}, {[j.name for j in self.routing.queue.items]}")
            ### 1. Tardy job before process ###
            for job in self.routing.queue.items:
                if job.due_date < now:
                    reward -= now - max(job.due_date, self.last_decision_time)
                    # print(f"1. {job.name}, reward {now - max(job.due_date, self.last_decision_time), [j.name for j in self.routing.queue.items]}")
            ### 2. Tardy job processing ###
            for i in range(self.num_machine):
                machine = self.process_dict[f"Machine {i}"]
                if not machine.idle:
                    job = machine.job_process
                    if now > job.due_date:
                        reward -= now - max(job.due_date, self.last_decision_time)
#                         print(f"2. {job.name} : reward {now - max(job.due_date, self.last_decision_time)}")

            ### 3. Tardy job end at now ###
            for job in self.sink.job_list:
                if job.sink_just:
                    if job.due_date < job.completion_time:
                        reward -= job.completion_time - max(job.due_date, self.last_decision_time)
#                         print(f"3. {job.name} : reward {job.completion_time - max(job.due_date, self.last_decision_time)}")
                        # print(f"{job.name} now:{job.completion_time}, dd:{job.due_date}, last_decision_time:{self.last_decision_time}")
                        # print(f"@: {job.completion_time - max(job.due_date, self.last_decision_time)}")
                    job.sink_just = False
#             print(f"@ now: {now} reward:{reward}")
            self.last_decision_time = now
        return reward

    def _get_state(self) -> np.ndarray:
        """
        feature 1 (U_ave): Avg of machine utilization rate
        feature 2 (U_std): Std deviation of machine utilization rate
        feature 3 (CRO_ave): Avg of operation completion rate
        feature 4 (CRJ_ave): Avg of job completion rate
        feature 5 (CRJ_std): Std deviation of job completion rate
        feature 6 (Tard_e): Estimated tardiness rate
        feature 7 (Tard_a): Actual tardiness rate
        """
        utilization_rate = [m.utilization_rate() for m in self.process_dict.values()]
        f_1 = np.mean(utilization_rate)
        f_2 = np.std(utilization_rate)
        temp_3 = 0
        CRJ = []
        num_ops = 0
        for j in self.jt_dict.values():
            temp_3 += j.num_completed_ops()
            num_ops += j.num_ops()
            CRJ.append(j.completion_rate())
        f_3 = temp_3 / num_ops
        f_4 = np.mean(CRJ)
        f_5 = np.std(CRJ)

        # # feature 1 (U_ave)
        # utilization_rate = [m.utilization_rate() for m in self.process_dict.values()]
        # U_ave = np.mean(utilization_rate)

        # feature 6 (Tard_e)
        T_cur = np.mean([m.completion_time_last_op() for m in self.process_dict.values()])  # avg of CT_k
        N_tard = 0
        N_left = 0
        for j in self.jt_dict.values():
            if j.num_completed_ops() < j.num_ops():
                N_left += j.num_ops() - j.num_completed_ops()
                T_left = 0
                idx = j.num_completed_ops()
                for left_op in j.left_op:
                    idx += 1
                    t_bar = np.mean(list(left_op.processing_time.values()))
                    T_left += t_bar
                    if T_cur + T_left > j.due_date:
                        N_tard += (j.num_ops() - idx + 1)
                        break
        Tard_e = N_tard / N_left if N_left != 0 else 0  # exception handling at final state return
        f_6 = Tard_e  # _calculate_reward 에서 계산함

        # feature 7 (Tard_a)
        N_tard = 0
        N_left = 0
        for j in self.jt_dict.values():
            if j.num_completed_ops() < j.num_ops():
                N_left += j.num_ops() - j.num_completed_ops()
                if j.completion_time > j.due_date:
                    N_tard += j.num_ops() - j.num_completed_ops()
        Tard_a = N_tard / N_left if N_left != 0 else 0  # exception handling at final state return
        f_7 = Tard_a  # _calculate_reward 에서 계산함

        state = np.concatenate((f_1, f_2, f_3, f_4, f_5, f_6, f_7), axis=None)
        return state

    def _generating_data(self) -> Tuple[list, list]:

        processing_time = [[[np.random.randint(1, 50) for _ in range(self.num_machine)]
                            for _ in range(np.random.randint(1, 20))]
                           for _ in range(self.num_job)]
        p_ij = []
        for i in range(self.num_job):
            p_ij_temp = []
            for j in range(len(processing_time[i])):
                num_avail_machine = np.random.randint(1, self.num_machine)
                l = len(processing_time[i][j])
                rand_idx = random.sample(range(l), self.num_machine - num_avail_machine)
                for idx in rand_idx:
                    processing_time[i][j][idx] = -1
                temp_list = [x for x in processing_time[i][j] if x != -1]
                p_ij_temp.append(np.mean(temp_list))
            p_ij.append(p_ij_temp)
        return processing_time, p_ij
