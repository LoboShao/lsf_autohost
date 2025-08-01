import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import deque, defaultdict


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: int
    cores_required: int
    memory_required: int  # in GB
    duration: int  # in seconds
    arrival_time: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: JobStatus = JobStatus.PENDING
    assigned_host: Optional[int] = None
    attempts: int = 0
    max_attempts: int = 3


@dataclass
class Host:
    id: int
    total_cores: int
    total_memory: int  # in GB
    available_cores: int
    available_memory: int
    running_job_ids: Set[int] = field(default_factory=set)  # Just store IDs

    def can_accommodate(self, job: Job) -> bool:
        return (self.available_cores >= job.cores_required and
                self.available_memory >= job.memory_required)

    def allocate_job(self, job: Job) -> bool:
        if self.can_accommodate(job):
            self.available_cores -= job.cores_required
            self.available_memory -= job.memory_required
            self.running_job_ids.add(job.id)
            job.assigned_host = self.id
            job.status = JobStatus.RUNNING
            return True
        return False

    def release_job(self, job: Job):
        if job.id in self.running_job_ids:
            self.available_cores += job.cores_required
            self.available_memory += job.memory_required
            self.running_job_ids.remove(job.id)
            job.assigned_host = None


class ClusterSchedulerEnv:
    def __init__(self,
                 num_hosts: int = 1000,
                 max_queue_length: int = 500,
                 # Host configuration ranges
                 host_cores_range: Tuple[int, int] = (8, 64),
                 host_memory_range: Tuple[int, int] = (16, 256),  # in GB
                 # Job generation parameters
                 job_cores_range: Tuple[int, int] = (1, 16),
                 job_memory_range: Tuple[int, int] = (1, 32),  # in GB
                 job_duration_range: Tuple[int, int] = (1, 60),  # in seconds
                 max_jobs_per_step: int = 50,  # Maximum jobs that can arrive per timestep
                 max_jobs_scheduled_per_step: int = 20,  # Maximum jobs to schedule per step
                 # Environment parameters
                 episode_length: int = 300,  # seconds
                 failure_probability: float = 0.02,
                 # Batch scheduling parameters (for LSF plugin modeling)
                 batch_window_seconds: float = 5.0,  # Time window to collect jobs before scheduling
                 min_batch_size: int = 1,  # Minimum jobs to trigger scheduling cycle
                 seed: Optional[int] = None):

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.num_hosts = num_hosts
        self.max_queue_length = max_queue_length
        self.host_cores_range = host_cores_range
        self.host_memory_range = host_memory_range
        self.job_cores_range = job_cores_range
        self.job_memory_range = job_memory_range
        self.job_duration_range = job_duration_range
        self.max_jobs_per_step = max_jobs_per_step
        self.max_jobs_scheduled_per_step = max_jobs_scheduled_per_step
        self.episode_length = episode_length
        self.failure_probability = failure_probability
        self.batch_window_seconds = batch_window_seconds
        self.min_batch_size = min_batch_size

        # Pre-allocate arrays for efficiency
        self.host_core_utils = np.zeros(num_hosts, dtype=np.float32)
        self.host_memory_utils = np.zeros(num_hosts, dtype=np.float32)

        # Initialize hosts
        self.hosts = self._create_hosts()

        # Use deque for O(1) queue operations
        self.job_queue = deque()

        # Efficient job tracking
        self.active_jobs = {}  # job_id -> Job
        self.completion_heap = []  # (completion_time, job_id)

        # Environment state
        self.current_time = 0.0
        self.next_job_id = 0

        # Scheduling cycle tracking
        self.scheduling_cycle_start_time = 0.0
        self.jobs_to_schedule_this_cycle = deque()  # Jobs queued for this scheduling cycle
        self.scheduling_cycle_active = False
        self.last_cycle_time = 0.0  # Track when last cycle started

        # Pre-allocated job generation
        self.job_batch_size = max(1, max_jobs_per_step * 3)  # Pre-generate 3 timesteps worth
        self.pending_jobs = deque()

        # Metrics (use simple counters for efficiency)
        self.total_jobs_generated = 0
        self.total_jobs_completed = 0
        self.total_jobs_failed = 0
        self.total_wait_time = 0.0
        self.total_turnaround_time = 0.0

    def _create_hosts(self) -> List[Host]:
        # Vectorized host creation
        cores = np.random.randint(self.host_cores_range[0], self.host_cores_range[1] + 1, self.num_hosts)
        memory = np.random.randint(self.host_memory_range[0], self.host_memory_range[1] + 1, self.num_hosts)

        hosts = []
        for i in range(self.num_hosts):
            hosts.append(Host(
                id=i,
                total_cores=int(cores[i]),
                total_memory=int(memory[i]),
                available_cores=int(cores[i]),
                available_memory=int(memory[i])
            ))
        return hosts

    def _generate_job_batch(self, count: int) -> List[Job]:
        """Generate multiple jobs at once for efficiency"""
        cores = np.random.randint(self.job_cores_range[0], self.job_cores_range[1] + 1, count)
        memory = np.random.randint(self.job_memory_range[0], self.job_memory_range[1] + 1, count)
        durations = np.random.randint(self.job_duration_range[0], self.job_duration_range[1] + 1, count)

        jobs = []
        for i in range(count):
            job = Job(
                id=self.next_job_id,
                cores_required=int(cores[i]),
                memory_required=int(memory[i]),
                duration=int(durations[i]),
                arrival_time=self.current_time
            )
            self.next_job_id += 1
            jobs.append(job)

        self.total_jobs_generated += count
        return jobs

    def _replenish_job_batch(self):
        """Replenish the pending jobs batch when needed"""
        if len(self.pending_jobs) < self.job_batch_size // 4:
            new_jobs = self._generate_job_batch(self.job_batch_size)
            self.pending_jobs.extend(new_jobs)

    def _add_jobs_to_queue(self):
        """Add jobs based on variable arrival rate - 1 to max_jobs_per_step"""
        # Random number of jobs arrive each timestep (1 to max)
        num_new_jobs = np.random.randint(1, self.max_jobs_per_step + 1)

        # Ensure we have enough pre-generated jobs
        if len(self.pending_jobs) < num_new_jobs:
            self._replenish_job_batch()

        # Add jobs to queue (up to max capacity)
        jobs_to_add = min(num_new_jobs,
                          self.max_queue_length - len(self.job_queue),
                          len(self.pending_jobs))

        for _ in range(jobs_to_add):
            if self.pending_jobs:
                job = self.pending_jobs.popleft()
                job.arrival_time = self.current_time  # Update arrival time
                self.job_queue.append(job)

    def _process_completions(self):
        """Process job completions - optimized with heap"""
        completed_jobs = []

        while self.completion_heap and self.completion_heap[0][0] <= self.current_time:
            completion_time, job_id = heapq.heappop(self.completion_heap)

            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job.end_time = completion_time
                job.status = JobStatus.COMPLETED

                # Release from host
                if job.assigned_host is not None:
                    self.hosts[job.assigned_host].release_job(job)

                # Update metrics
                wait_time = job.start_time - job.arrival_time
                turnaround_time = job.end_time - job.arrival_time
                self.total_wait_time += wait_time
                self.total_turnaround_time += turnaround_time
                self.total_jobs_completed += 1

                completed_jobs.append(job)
                del self.active_jobs[job_id]

        return completed_jobs

    def _update_host_utilization(self):
        """Update host utilization arrays efficiently"""
        for i, host in enumerate(self.hosts):
            self.host_core_utils[i] = 1.0 - (host.available_cores / host.total_cores)
            self.host_memory_utils[i] = 1.0 - (host.available_memory / host.total_memory)

    def get_state(self) -> np.ndarray:
        """Get current environment state - simplified for understanding"""
        # Update utilization arrays
        self._update_host_utilization()

        # Core state: host utilizations (cores and memory for each host)
        state_parts = [
            self.host_core_utils,  # [1000] - core utilization per host
            self.host_memory_utils  # [1000] - memory utilization per host
        ]

        # Queue info for scheduling cycle
        queue_length = len(self.jobs_to_schedule_this_cycle) if self.scheduling_cycle_active else len(self.job_queue)
        queue_util = queue_length / self.max_queue_length

        # Next job info (only what scheduler knows: cores and memory)
        if self.scheduling_cycle_active and self.jobs_to_schedule_this_cycle:
            next_job = self.jobs_to_schedule_this_cycle[0]
            job_features = np.array([
                next_job.cores_required / self.job_cores_range[1],
                next_job.memory_required / self.job_memory_range[1],
            ], dtype=np.float32)
        elif not self.scheduling_cycle_active and self.job_queue:
            next_job = self.job_queue[0]
            job_features = np.array([
                next_job.cores_required / self.job_cores_range[1],
                next_job.memory_required / self.job_memory_range[1],
            ], dtype=np.float32)
        else:
            job_features = np.zeros(2, dtype=np.float32)

        # Simple global metrics
        avg_core_util = np.mean(self.host_core_utils)
        avg_memory_util = np.mean(self.host_memory_utils)

        global_features = np.array([
            queue_util,  # Queue pressure
            avg_core_util,  # Overall core usage
            avg_memory_util,  # Overall memory usage
            float(self.scheduling_cycle_active),  # Whether we're in a scheduling cycle
        ], dtype=np.float32)

        # Combine all features
        state_parts.extend([job_features, global_features])
        return np.concatenate(state_parts)

    def _create_job_buckets(self, jobs: List[Job]) -> Dict[str, List[Job]]:
        """Group jobs into buckets based on identical resource requirements (LSF-like)"""
        buckets = defaultdict(list)

        for job in jobs:
            # Create bucket key based on resource requirements only
            bucket_key = f"{job.cores_required}c_{job.memory_required}m"
            buckets[bucket_key].append(job)

        return buckets

    def _schedule_job_batch(self, action: np.ndarray, max_jobs: int):
        """Schedule jobs using LSF-like bucket strategy with skip optimization"""
        if not self.jobs_to_schedule_this_cycle:
            return 0, 0, 0.0

        jobs_scheduled = 0
        jobs_failed = 0
        total_reward = 0.0

        # Sort hosts by priority once (descending order)
        host_priorities = np.argsort(action)[::-1]

        # Get jobs to process in FIFO order (first max_jobs from cycle queue)
        jobs_to_process = []
        for _ in range(min(max_jobs, len(self.jobs_to_schedule_this_cycle))):
            if self.jobs_to_schedule_this_cycle:
                jobs_to_process.append(self.jobs_to_schedule_this_cycle.popleft())

        # Create buckets from jobs we're about to process
        job_buckets = self._create_job_buckets(jobs_to_process)

        # Track which jobs couldn't be scheduled (will go back to cycle queue)
        unscheduled_jobs = []

        # Process each bucket
        for bucket_key, bucket_jobs in job_buckets.items():
            bucket_can_schedule = True  # Track if this bucket type can still schedule

            for job in bucket_jobs:
                # LSF optimization: if bucket already failed, skip remaining jobs in bucket
                if not bucket_can_schedule:
                    unscheduled_jobs.append(job)
                    continue

                scheduled = False

                # Try hosts in priority order
                for host_idx in host_priorities:
                    host = self.hosts[host_idx]

                    if host.can_accommodate(job):
                        # Simulate failure
                        if np.random.random() < self.failure_probability:
                            job.attempts += 1
                            if job.attempts >= job.max_attempts:
                                job.status = JobStatus.FAILED
                                jobs_failed += 1
                                self.total_jobs_failed += 1
                                total_reward -= 10.0
                                bucket_can_schedule = False  # Mark bucket as failed
                                break
                            continue

                        # Successfully schedule
                        if host.allocate_job(job):
                            job.start_time = self.scheduling_cycle_start_time
                            self.active_jobs[job.id] = job

                            # Schedule completion based on cycle start time
                            completion_time = self.scheduling_cycle_start_time + job.duration
                            heapq.heappush(self.completion_heap, (completion_time, job.id))

                            scheduled = True
                            jobs_scheduled += 1

                            # Simple reward: +1 for each scheduled job
                            total_reward += 1.0
                            break

                if not scheduled:
                    # Job couldn't be scheduled
                    if job.attempts == 0:
                        bucket_can_schedule = False  # First failure in bucket - mark bucket as unable
                    unscheduled_jobs.append(job)

        # Put unscheduled jobs back at the front of cycle queue (LSF behavior)
        for job in reversed(unscheduled_jobs):  # Reverse to maintain order
            self.jobs_to_schedule_this_cycle.appendleft(job)

        return jobs_scheduled, jobs_failed, total_reward

    def start_scheduling_cycle(self):
        """Start a new scheduling cycle by moving jobs from main queue to scheduling queue"""
        if self.scheduling_cycle_active:
            return False  # Already in a cycle

        if not self.job_queue:
            return False  # No jobs to schedule

        # Start new scheduling cycle
        self.scheduling_cycle_active = True
        self.scheduling_cycle_start_time = self.current_time

        # Move all jobs from main queue to scheduling cycle queue
        while self.job_queue:
            self.jobs_to_schedule_this_cycle.append(self.job_queue.popleft())

        return True

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one scheduling step within a cycle"""
        # Validate action dimensions
        if len(action) != self.num_hosts:
            raise ValueError(f"Action must have {self.num_hosts} elements (one per host), got {len(action)}")

        # If no active cycle, check if we should start one based on batch conditions
        if not self.scheduling_cycle_active:
            # Start cycle if: 1) Time window elapsed OR 2) Enough jobs queued
            time_elapsed = self.current_time - self.last_cycle_time >= self.batch_window_seconds
            enough_jobs = len(self.job_queue) >= self.min_batch_size
            
            if self.job_queue and (time_elapsed or enough_jobs):
                self.start_scheduling_cycle()
                self.last_cycle_time = self.current_time  # Update last cycle time
            
            if not self.scheduling_cycle_active:  # Still not active means no jobs to schedule
                # No jobs to schedule, just advance time and add new jobs
                self._add_jobs_to_queue()
                self.current_time += 1.0

                info = {
                    'jobs_scheduled': 0,
                    'queue_length': len(self.job_queue),
                    'active_jobs': len(self.active_jobs),
                    'cycle_completed': False
                }

                done = self.current_time >= self.episode_length
                return self.get_state(), 0.0, done, info

        # Schedule jobs within the active cycle
        total_reward = 0.0
        jobs_scheduled = 0
        jobs_failed = 0

        # Always try to schedule if we have jobs in the cycle queue
        if self.jobs_to_schedule_this_cycle:
            jobs_scheduled, jobs_failed, reward = self._schedule_job_batch(
                action, max_jobs=self.max_jobs_scheduled_per_step
            )
            total_reward += reward

        # Check if scheduling cycle is complete
        # Cycle completes when either:
        # 1. All jobs scheduled successfully (queue empty)
        # 2. No jobs were scheduled in this step (indicating insufficient resources)
        cycle_completed = (len(self.jobs_to_schedule_this_cycle) == 0 or
                           (len(self.jobs_to_schedule_this_cycle) > 0 and jobs_scheduled == 0))

        if cycle_completed:
            # End scheduling cycle - jobs start running and time advances
            self.scheduling_cycle_active = False

            # If we have unscheduled jobs, put them back in main queue
            if self.jobs_to_schedule_this_cycle:
                # Move remaining jobs back to main queue for next cycle
                while self.jobs_to_schedule_this_cycle:
                    self.job_queue.appendleft(self.jobs_to_schedule_this_cycle.pop())

            # Process any completions that happened during the cycle
            self._process_completions()

            # Add new jobs for next cycle
            self._add_jobs_to_queue()

            # Advance time by 1 second
            self.current_time += 1.0

            # Process completions again after time advance
            self._process_completions()

        # Simple additional rewards/penalties
        queue_penalty = (len(self.job_queue) + len(self.jobs_to_schedule_this_cycle)) / self.max_queue_length
        total_reward -= queue_penalty  # Penalty for long queues

        avg_core_util = np.mean(self.host_core_utils)
        avg_memory_util = np.mean(self.host_memory_utils)
        if cycle_completed:
            total_reward += avg_core_util  # Reward for high utilization only when cycle completes

        # Check completion
        done = self.current_time >= self.episode_length

        info = {
            'jobs_scheduled': jobs_scheduled,
            'queue_length': len(self.job_queue),
            'active_jobs': len(self.active_jobs),
            'cycle_completed': cycle_completed
        }

        return self.get_state(), total_reward, done, info

    def reset(self) -> np.ndarray:
        """Reset environment - optimized"""
        # Reset hosts efficiently
        for host in self.hosts:
            host.available_cores = host.total_cores
            host.available_memory = host.total_memory
            host.running_job_ids.clear()

        # Clear collections
        self.job_queue.clear()
        self.jobs_to_schedule_this_cycle.clear()
        self.active_jobs.clear()
        self.completion_heap.clear()
        self.pending_jobs.clear()

        # Reset state
        self.current_time = 0.0
        self.next_job_id = 0
        self.scheduling_cycle_active = False
        self.scheduling_cycle_start_time = 0.0
        self.last_cycle_time = 0.0

        # Reset metrics
        self.total_jobs_generated = 0
        self.total_jobs_completed = 0
        self.total_jobs_failed = 0
        self.total_wait_time = 0.0
        self.total_turnaround_time = 0.0

        # Reset utilization arrays
        self.host_core_utils.fill(0.0)
        self.host_memory_utils.fill(0.0)

        # Pre-generate initial job batch
        self._replenish_job_batch()

        return self.get_state()

    @property
    def action_space(self):
        class ActionSpace:
            def __init__(self, num_hosts):
                self.shape = (num_hosts,)
                self.low = 0.0
                self.high = 1.0
            def sample(self):
                import numpy as np
                return np.random.random(self.shape)
        return ActionSpace(self.num_hosts)

    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        total_jobs = max(1, self.total_jobs_generated)
        completed_jobs = max(1, self.total_jobs_completed)

        return {
            'total_jobs_generated': self.total_jobs_generated,
            'total_jobs_completed': self.total_jobs_completed,
            'total_jobs_failed': self.total_jobs_failed,
            'active_jobs': len(self.active_jobs),
            'queue_length': len(self.job_queue),
            'cycle_queue_length': len(self.jobs_to_schedule_this_cycle),
            'completion_rate': self.total_jobs_completed / total_jobs,
            'failure_rate': self.total_jobs_failed / total_jobs,
            'avg_wait_time': self.total_wait_time / completed_jobs,
            'avg_turnaround_time': self.total_turnaround_time / completed_jobs,
            'avg_host_core_utilization': np.mean(self.host_core_utils),
            'avg_host_memory_utilization': np.mean(self.host_memory_utils),
            'throughput': self.total_jobs_completed / max(1, self.current_time),
            'scheduling_cycle_active': self.scheduling_cycle_active
        }


# Quick performance test
if __name__ == "__main__":
    import time
    print("Testing single-step reward logic for Python ClusterSchedulerEnv...")
    env = ClusterSchedulerEnv(
        num_hosts=100,
        max_queue_length=50,
        episode_length=20,
        seed=42
    )
    obs = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    total_reward = 0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {step+1}: reward={reward}")
        total_reward += reward
        if done:
            break
    print(f"Total reward: {total_reward}")
    print(f"Final metrics: {env.get_metrics()}")