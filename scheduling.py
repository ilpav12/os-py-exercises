import copy
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from enum import Enum
from matplotlib.colors import ListedColormap


class State(Enum):
    NONE = 0
    RUNNING = 1
    READY = 2
    WAITING = 3
    CS = 4


def get_state_names():
    return [state.name for state in State]

def get_state_values():
    return [state.value for state in State]


class Job:
    def __init__(self, name: str, duration: int, starting_time: int, priority: int = 1,
                 finishing_time: int = None,
                 response_time: int = None, completion_time: int = None, waiting_time: int = 0):
        self.name = name
        self.duration = duration
        self.remaining_time = duration
        self.starting_time = starting_time
        self.priority = priority
        self.finishing_time = finishing_time
        self.response_time = response_time
        self.completion_time = completion_time
        self.waiting_time = waiting_time
        self.states = []

    def __str__(self):
        return str(self.to_array())

    def to_array(self):
        return [self.name, self.duration, self.starting_time, self.finishing_time, self.response_time,
                self.completion_time, self.waiting_time]

    def get_states_values(self):
        return [state.value for state in self.states]

    def get_num_state(self, state: State):
        return self.states.count(state)


class Jobs:
    def __init__(self, jobs: list[Job]):
        self.jobs = jobs

    def __iter__(self):
        return iter(self.jobs)

    def __getitem__(self, index):
        return self.jobs[index]

    def __len__(self):
        return len(self.jobs)

    def __str__(self):
        return str(self.to_array())

    def __repr__(self):
        return str(self.to_array())

    def append(self, job: Job):
        self.jobs.append(job)

    def remove(self, job: Job):
        self.jobs.remove(job)

    def copy(self):
        return Jobs([job for job in self.jobs])

    def deepcopy(self):
        return Jobs([copy.deepcopy(job) for job in self.jobs])

    def to_array(self):
        return [job.to_array() for job in self.jobs]

    def to_pandas(self):
        return pd.DataFrame(self.to_array(),
                            columns=['Job', 'Duration', 'Starting Time', 'Finishing Time', 'Response Time',
                                     'Completion Time', 'Waiting Time']).set_index('Job')

    def sort_by_name(self, reverse: bool = False):
        return Jobs(sorted(self.jobs, key=lambda x: x.name, reverse=reverse))

    def avg_turnaround_time(self):
        return sum([job.completion_time for job in self.jobs]) / len(self.jobs)

    def avg_waiting_time(self):
        return sum([job.waiting_time for job in self.jobs]) / len(self.jobs)

    def avg_response_time(self):
        return sum([job.response_time for job in self.jobs]) / len(self.jobs)

    def throughput(self):
        last_job = max(self.jobs, key=lambda x: x.finishing_time)
        return len(self.jobs) / last_job.finishing_time


class SchedulingAlgorithm(ABC):
    def __init__(self, dispatch_latency: int = 0):
        self.dispatch_latency = dispatch_latency

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def process_jobs(self, jobs: Jobs):
        pass


class FCFS(SchedulingAlgorithm):
    def __init__(self, dispatch_latency: int = 0):
        super().__init__(dispatch_latency)

    def __str__(self):
        return f'FCFS with dispatch latency {self.dispatch_latency}ms'

    def process_jobs(self, jobs: Jobs):
        jobs_to_process = jobs.deepcopy()
        current_time = 0
        all_finished = False
        selected_job = None
        job_to_remove = None
        context_switch = self.dispatch_latency
        processed_jobs = Jobs([])
        ready_queue = Jobs([job for job in jobs_to_process if job.starting_time == current_time])
        while not all_finished:
            if len(jobs_to_process) == 0:
                all_finished = True
                continue

            # Select the job with the maximum waiting time if no job is selected
            if selected_job is None:
                selected_job = max(ready_queue, key=lambda x: x.waiting_time)
                selected_job.waiting_time = 0
                context_switch = self.dispatch_latency

            for job in jobs_to_process:
                if job.starting_time > current_time:  # Job has not arrived yet
                    job.states.append(State.NONE)
                    continue

                if job not in ready_queue:
                    ready_queue.append(job)

                if job != selected_job:
                    job.states.append(State.READY)
                    job.waiting_time += 1
                    continue

                if job.response_time is None:
                    job.response_time = current_time + self.dispatch_latency - job.starting_time

                if context_switch > 0:
                    job.states.append(State.CS)
                    context_switch -= 1
                    continue

                job.remaining_time -= 1
                job.states.append(State.RUNNING)

                if job.remaining_time == 0:
                    job.finishing_time = current_time + self.dispatch_latency
                    if self.dispatch_latency == 0: job.finishing_time += 1
                    job.completion_time = job.finishing_time - job.starting_time
                    job.waiting_time = job.completion_time - job.duration
                    processed_jobs.append(job)
                    job_to_remove = job
                    selected_job = None

            if job_to_remove is not None:
                jobs_to_process.remove(job_to_remove)
                job_to_remove = None

            current_time += 1

        return processed_jobs.sort_by_name()

class RR(SchedulingAlgorithm):
    def __init__(self, dispatch_latency: int = 0, quantum: int = 1):
        super().__init__(dispatch_latency)
        self.quantum = quantum

    def __str__(self):
        return f'RR with dispatch latency {self.dispatch_latency}ms and quantum {self.quantum}ms'

    def process_jobs(self, jobs: Jobs):
        jobs_to_process = jobs.deepcopy()
        current_time = 0
        all_finished = False
        change_selected_job = True
        selected_job = None
        job_to_remove = None
        context_switch = self.dispatch_latency
        processed_jobs = Jobs([])
        quantum = self.quantum
        while not all_finished:
            if len(jobs_to_process) == 0:
                all_finished = True
                continue

            ready_queue = Jobs([job for job in jobs_to_process if job.starting_time <= current_time])

            # Select the job with the maximum waiting time if no job is selected
            if change_selected_job:
                # filter most_waiting_jobs by taking only the ones with the highest priority
                highest_priority = min(ready_queue, key=lambda x: x.priority).priority
                highest_priority_jobs = Jobs([job for job in ready_queue if job.priority == highest_priority])
                most_waiting_jobs = Jobs([job for job in highest_priority_jobs if job.waiting_time == max(highest_priority_jobs, key=lambda x: x.waiting_time).waiting_time])
                new_selected_job = min(most_waiting_jobs, key=lambda x: x.get_num_state(State.RUNNING))
                if selected_job != new_selected_job:
                    context_switch = self.dispatch_latency
                selected_job = new_selected_job
                selected_job.waiting_time = 0
                change_selected_job = False

            for job in jobs_to_process:
                if job.starting_time > current_time:  # Job has not arrived yet
                    job.states.append(State.NONE)
                    continue

                if job != selected_job:
                    job.states.append(State.READY)
                    job.waiting_time += 1
                    continue

                if job.response_time is None:
                    job.response_time = current_time + self.dispatch_latency - job.starting_time

                if context_switch > 0:
                    job.states.append(State.CS)
                    context_switch -= 1
                    continue

                quantum -= 1
                job.remaining_time -= 1
                job.states.append(State.RUNNING)

                if job.remaining_time == 0:
                    job.finishing_time = current_time + self.dispatch_latency
                    if self.dispatch_latency == 0: job.finishing_time += 1
                    job.completion_time = job.finishing_time - job.starting_time
                    job.waiting_time = job.completion_time - job.duration
                    processed_jobs.append(job)
                    job_to_remove = job
                    change_selected_job = True
                    quantum = self.quantum

                if quantum == 0:
                    quantum = self.quantum
                    change_selected_job = True

            if job_to_remove is not None:
                jobs_to_process.remove(job_to_remove)
                job_to_remove = None

            current_time += 1

        return processed_jobs.sort_by_name()

class SJF(SchedulingAlgorithm):
    def __init__(self, dispatch_latency: int = 0):
        super().__init__(dispatch_latency)

    def __str__(self):
        return f'SJF with dispatch latency {self.dispatch_latency}ms'

    def process_jobs(self, jobs: Jobs):
        jobs_to_process = jobs.deepcopy()
        current_time = 0
        all_finished = False
        selected_job = None
        job_to_remove = None
        context_switch = self.dispatch_latency
        processed_jobs = Jobs([])
        while not all_finished:
            if len(jobs_to_process) == 0:
                all_finished = True
                continue

            # Select the job with the minimum remaining time if no job is selected
            if selected_job is None:
                jobs_to_select = [job for job in jobs_to_process if job.starting_time <= current_time]
                selected_job = min(jobs_to_select, key=lambda x: x.duration)
                context_switch = self.dispatch_latency

            for job in jobs_to_process:
                if job.starting_time > current_time:  # Job has not arrived yet
                    job.states.append(State.NONE)
                    continue

                if job != selected_job:
                    job.states.append(State.READY)
                    job.waiting_time += 1
                    continue

                if job.response_time is None:
                    job.response_time = current_time + self.dispatch_latency - job.starting_time

                if context_switch > 0:
                    job.states.append(State.CS)
                    context_switch -= 1
                    continue

                job.remaining_time -= 1
                job.states.append(State.RUNNING)

                if job.remaining_time == 0:
                    job.finishing_time = current_time + self.dispatch_latency
                    if self.dispatch_latency == 0: job.finishing_time += 1
                    job.completion_time = job.finishing_time - job.starting_time
                    job.waiting_time = job.completion_time - job.duration
                    processed_jobs.append(job)
                    job_to_remove = job
                    selected_job = None

            if job_to_remove is not None:
                jobs_to_process.remove(job_to_remove)
                job_to_remove = None

            current_time += 1

        return processed_jobs.sort_by_name()

class SRTF(SchedulingAlgorithm):
    def __init__(self, dispatch_latency: int = 0):
        super().__init__(dispatch_latency)

    def __str__(self):
        return f'SRTF with dispatch latency {self.dispatch_latency}ms'

    def process_jobs(self, jobs: Jobs):
        jobs_to_process = jobs.deepcopy()
        current_time = 0
        all_finished = False
        selected_job = None
        job_to_remove = None
        context_switch = self.dispatch_latency
        processed_jobs = Jobs([])
        while not all_finished:
            if len(jobs_to_process) == 0:
                all_finished = True
                continue

            # Select the job with the minimum remaining time if no job is selected
            jobs_to_select = [job for job in jobs_to_process if job.starting_time <= current_time]
            new_selected_job = min(jobs_to_select, key=lambda x: x.remaining_time)
            if selected_job != new_selected_job:
                context_switch = self.dispatch_latency
            selected_job = min(jobs_to_select, key=lambda x: x.remaining_time)

            for job in jobs_to_process:
                if job.starting_time > current_time:  # Job has not arrived yet
                    job.states.append(State.NONE)
                    continue

                if job != selected_job:
                    job.states.append(State.READY)
                    job.waiting_time += 1
                    continue

                if job.response_time is None:
                    job.response_time = current_time + self.dispatch_latency - job.starting_time

                if context_switch > 0:
                    job.states.append(State.CS)
                    context_switch -= 1
                    continue

                job.remaining_time -= 1
                job.states.append(State.RUNNING)

                if job.remaining_time == 0:
                    job.finishing_time = current_time + self.dispatch_latency
                    if self.dispatch_latency == 0: job.finishing_time += 1
                    job.completion_time = job.finishing_time - job.starting_time
                    job.waiting_time = job.completion_time - job.duration
                    processed_jobs.append(job)
                    job_to_remove = job

            if job_to_remove is not None:
                jobs_to_process.remove(job_to_remove)
                job_to_remove = None

            current_time += 1

        return processed_jobs.sort_by_name()

class HRRF(SchedulingAlgorithm):
    def __init__(self, dispatch_latency: int = 0):
        super().__init__(dispatch_latency)

    def __str__(self):
        return f'HRRF with dispatch latency {self.dispatch_latency}ms'

    def process_jobs(self, jobs: Jobs):
        jobs_to_process = jobs.deepcopy()
        current_time = 0
        all_finished = False
        selected_job = None
        job_to_remove = None
        context_switch = self.dispatch_latency
        processed_jobs = Jobs([])
        while not all_finished:
            if len(jobs_to_process) == 0:
                all_finished = True
                continue

            # Select the job with the minimum remaining time if no job is selected
            if selected_job is None:
                jobs_to_select = [job for job in jobs_to_process if job.starting_time <= current_time]
                new_selected_job = max(jobs_to_select, key=lambda x: (x.duration + x.waiting_time) / x.duration)
                if selected_job != new_selected_job:
                    context_switch = self.dispatch_latency
                selected_job = new_selected_job

            for job in jobs_to_process:
                if job.starting_time > current_time:  # Job has not arrived yet
                    job.states.append(State.NONE)
                    continue

                if job != selected_job:
                    job.states.append(State.READY)
                    job.waiting_time += 1
                    continue

                if job.response_time is None:
                    job.response_time = current_time + self.dispatch_latency - job.starting_time

                if context_switch > 0:
                    job.states.append(State.CS)
                    context_switch -= 1
                    continue

                job.remaining_time -= 1
                job.states.append(State.RUNNING)

                if job.remaining_time == 0:
                    job.finishing_time = current_time + self.dispatch_latency
                    if self.dispatch_latency == 0: job.finishing_time += 1
                    job.completion_time = job.finishing_time - job.starting_time
                    job.waiting_time = job.completion_time - job.duration
                    processed_jobs.append(job)
                    job_to_remove = job
                    selected_job = None

            if job_to_remove is not None:
                jobs_to_process.remove(job_to_remove)
                job_to_remove = None

            current_time += 1

        return processed_jobs.sort_by_name()


def plot_gantt_chart(jobs: Jobs):
    job_state_names = get_state_names()
    # create custom colormap with custom colors for each state using a dictionary
    cmap = ListedColormap(['white', 'red', 'blue', 'yellow', 'gray'])


    x, y, c = [], [], []
    for i, job in enumerate(reversed(jobs)):
        x += [x + 0.5 for x in range(len(job.get_states_values()))]
        y += [i] * len(job.get_states_values())
        c += job.get_states_values()

    fig, ax = plt.subplots(figsize=(15, (len(jobs) + 1)))
    ax.set_yticks(range(len(jobs)))
    ax.set_yticklabels([job.name for job in reversed(jobs)])
    ax.set_xticks(range(0, round(max(x)) + 2, 1))
    ax.set_xlabel('Time')
    ax.set_ylabel('Job')
    ax.grid(which='both', axis='x', linestyle='-', color='black', alpha=0.25)
    ax.scatter(
        x, y,
        c=c,
        marker='_',
        lw=20,
        cmap=cmap,
        vmin=0, vmax=len(job_state_names) - 1,
    )
    # make color legend for each state
    handles = [plt.plot([], [], c=cmap(i), lw=10, marker="_")[0] for i in range(len(job_state_names))]
    labels = job_state_names
    # put legend outside the plot on the bottom
    plt.legend(handles, labels, loc=(0.2, -0.3), numpoints=1, fontsize=12,
               ncol=len(job_state_names))
    plt.show()