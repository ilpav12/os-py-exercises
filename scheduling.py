from abc import ABC, abstractmethod

class Job:
    def __init__(self, duration: int, starting_time: int, finishing_time: int = None,
                 response_time: int = None, completion_time: int = None, waiting_time: int = None):
        self.duration = duration
        self.starting_time = starting_time
        self.finishing_time = finishing_time
        self.response_time = response_time
        self.completion_time = completion_time
        self.waiting_time = waiting_time

    def __str__(self):
        return print(self.to_array())

    def to_array(self):
        return [self.duration, self.starting_time, self.finishing_time, self.response_time,
                self.completion_time, self.waiting_time]


class Jobs:
    def __init__(self, jobs: list[Job]):
        self.jobs = jobs

    def __str__(self):
        if not self.jobs:
            return "No jobs to display"

        header = ['Duration', 'Starting Time', 'Finishing Time', 'Response Time', 'Completion Time', 'Waiting Time']
        return print(header, [job.to_array() for job in self.jobs])

    def __repr__(self):
        header = ['Duration', 'Starting Time', 'Finishing Time', 'Response Time', 'Completion Time', 'Waiting Time']
        return [header, [job.to_array() for job in self.jobs]]

    def to_array(self):
        return [job.to_array() for job in self.jobs]


class SchedulingAlgorithm(ABC):
    def __init__(self, dispatch_latency: int = 0):
        self.dispatch_latency = dispatch_latency

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def process_jobs(self, jobs: list[Job]):
        pass


class FCFS(SchedulingAlgorithm):
    def __init__(self, dispatch_latency: int = 0):
        super().__init__(dispatch_latency)

    def __str__(self):
        return f'FCFS with dispatch latency {self.dispatch_latency}ms'

    def process_jobs(self, jobs: list[Job]):
        current_time = 0
        for job in jobs:
            job.finishing_time = current_time + job.duration + self.dispatch_latency
            job.response_time = current_time + self.dispatch_latency - job.starting_time
            job.completion_time = job.finishing_time - job.starting_time
            job.waiting_time = job.completion_time - job.duration
            current_time = job.finishing_time

        return jobs
