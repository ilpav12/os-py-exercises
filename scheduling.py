import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from enum import Enum
from matplotlib.colors import ListedColormap


class States(Enum):
    NONE = 0
    RUNNING = 1
    READY = 2
    WAITING = 3
    CS = 4


class Job:
    def __init__(self, name: str, duration: int, starting_time: int, finishing_time: int = None,
                 response_time: int = None, completion_time: int = None, waiting_time: int = 0):
        self.name = name
        self.duration = duration
        self.remaining_time = duration
        self.starting_time = starting_time
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


class Jobs:
    def __init__(self, jobs: list[Job]):
        self.jobs = jobs

    def __iter__(self):
        return iter(self.jobs)

    def __getitem__(self, index):
        return self.jobs[index]

    def append(self, job: Job):
        self.jobs.append(job)

    def remove(self, job: Job):
        self.jobs.remove(job)

    def __len__(self):
        return len(self.jobs)

    def __str__(self):
        return str(self.to_array())

    def __repr__(self):
        return str(self.to_array())

    def to_array(self):
        return [job.to_array() for job in self.jobs]

    def to_pandas(self):
        return pd.DataFrame(self.to_array(),
                            columns=['Job', 'Duration', 'Starting Time', 'Finishing Time', 'Response Time',
                                     'Completion Time', 'Waiting Time']).set_index('Job')

    def sort_by_name(self, reverse: bool = False):
        return Jobs(sorted(self.jobs, key=lambda x: x.name, reverse=reverse))


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
        current_time = 0
        all_finished = False
        selected_job = None
        job_to_remove = None
        context_switch = self.dispatch_latency
        processed_jobs = Jobs([])
        while not all_finished:
            if len(jobs) == 0:
                all_finished = True
                continue

            # Select the job with the maximum waiting time if no job is selected
            if selected_job is None:
                selected_job = max(jobs, key=lambda x: x.waiting_time)
                context_switch = self.dispatch_latency

            for job in jobs:
                if job.starting_time > current_time:  # Job has not arrived yet
                    job.states.append(States.NONE)
                    continue

                if job != selected_job:
                    job.states.append(States.READY)
                    job.waiting_time += 1
                    continue

                if job.response_time is None:
                    job.response_time = current_time + self.dispatch_latency - job.starting_time

                if context_switch > 0:
                    job.states.append(States.CS)
                    context_switch -= 1
                    continue

                job.remaining_time -= 1
                job.states.append(States.RUNNING)

                if job.remaining_time == 0:
                    job.finishing_time = current_time + self.dispatch_latency
                    if self.dispatch_latency == 0: job.finishing_time += 1
                    job.completion_time = job.finishing_time - job.starting_time
                    job.waiting_time = job.completion_time - job.duration
                    processed_jobs.append(job)
                    job_to_remove = job
                    selected_job = None

            if job_to_remove is not None:
                jobs.remove(job_to_remove)
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
        current_time = 0
        all_finished = False
        change_selected_job = True
        selected_job = None
        job_to_remove = None
        context_switch = self.dispatch_latency
        processed_jobs = Jobs([])
        quantum = self.quantum
        while not all_finished:
            if len(jobs) == 0:
                all_finished = True
                continue

            # Select the job with the maximum waiting time if no job is selected
            if change_selected_job:
                new_selected_job = max(jobs, key=lambda x: x.waiting_time)
                if selected_job != new_selected_job:
                    context_switch = self.dispatch_latency
                selected_job = new_selected_job
                change_selected_job = False

            for job in jobs:
                if job.starting_time > current_time:  # Job has not arrived yet
                    job.states.append(States.NONE)
                    continue

                if job != selected_job:
                    job.states.append(States.READY)
                    job.waiting_time += 1
                    continue

                if job.response_time is None:
                    job.response_time = current_time + self.dispatch_latency - job.starting_time

                if context_switch > 0:
                    job.states.append(States.CS)
                    context_switch -= 1
                    continue

                quantum -= 1
                job.remaining_time -= 1
                job.states.append(States.RUNNING)

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
                jobs.remove(job_to_remove)
                job_to_remove = None

            current_time += 1

        return processed_jobs.sort_by_name()

def plot_gantt_chart(jobs: Jobs):
    jobs_states = [state.name for state in States]
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
        vmin=0, vmax=len(jobs_states) - 1,
    )
    # make color legend for each state
    handles = [plt.plot([], [], c=cmap(i), lw=10, marker="_")[0] for i in range(len(jobs_states))]
    labels = jobs_states
    # put legend outside the plot on the bottom
    plt.legend(handles, labels, loc=(0.2, -0.3), numpoints=1, fontsize=12,
               ncol=len(jobs_states))
    plt.show()