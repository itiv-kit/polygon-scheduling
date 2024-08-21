import math
from typing import Any
import numpy
import scipy.optimize
import platform_exports
import platform_exports.platform_tools
from platform_exports.task_tools import task
import platform_exports.task_tools
import polyhedron_heuristics.schedule_tools
import scipy

TASK_OVERLAP_PENALTY = 3000
ACCESS_WINDOW_PENALTY = 500
SEC_WINDOW_PENALTY = 750
RUNTIME_OVERSHOOT_PENALTY = 2000
DEADLINE_MISS_PENALTY = 1500
MULTIPLE_EXEC_COST = 250

NO_EXEC_SEVERITY = 100

UNUSED_START = -1

MAX_ITERATIONS = 2000

PLATFORM_MINIMUM_RES = 100


class soft_cost_function:
    """
    Cost function class that gets called to determine the penaltys and costs of a proposed schedule.

    Members:
        tasks_: list[task] - A list of all tasks used for indexing
        no_units_: int - The number of different execution units
        max_runtime_: int = math.inf - The maximum allowed runtime
        max_exex_number_: int = 1 - the number of allowed executions per TDMA schedule table

    Definition of functions is based on optimizing the vector x : numpy.array, which is build like this:
    [t_00, t_00, ..., t_10, t_11, ..., u_00, u_01, ..., u_10, u_11, ...]
    At first, the time of every execution is presented.
    Afterward, execution unit for each starting point is presented.

    If a start is not used, this is determined by the UNUSED_START definition.
    """

    def __init__(
        self,
        tasks: list[task],
        no_units: int,
        max_runtime: int = math.inf,
        max_exec_number: int = 1,
    ) -> None:
        self.tasks_ = tasks
        self.no_units_ = no_units
        self.max_runtime_ = max_runtime
        self.max_exec_number_ = max_exec_number

    def _sorted_overlapping_time(self, start1: int, start2: int, end1: int, end2: int) -> int:
        """
        Return the overlap between sorted time slots.
        """
        return len(range(max(start1, start2), min(end1, end2)))

    def _overlapping_time(self, start_time1: int, start_time2: int, end_time1: int, end_time2: int) -> int:
        """
        Calculates the overlapping time slots between two (1 and 2) time periods and account for possible rollover.
        """
        overlap = 0

        # Check, if execution rolls around
        rollover1 = start_time1 > end_time1
        rollover2 = start_time2 > end_time2

        # If both roll over, calculate overlap as t_max - t_end_max + t_start_min (but still using functions)
        if rollover1 and rollover2:
            overlap_start = self._sorted_overlapping_time(0, 0, end_time1, end_time2)
            overlap_end = self._sorted_overlapping_time(start_time1, start_time2, self.max_runtime_, self.max_runtime_)
            overlap += overlap_start + overlap_end

        # If only one task rolls over, calculate start and end respectively (only one should trigger)
        elif not rollover1 and rollover2:
            overlap_start = self._sorted_overlapping_time(start_time1, 0, end_time1, end_time2)
            overlap_end = self._sorted_overlapping_time(start_time1, start_time2, end_time1, self.max_runtime_)
            overlap += overlap_start + overlap_end

        # Same as above
        elif rollover1 and not rollover2:
            overlap_start = self._sorted_overlapping_time(0, start_time2, end_time1, end_time2)
            overlap_end = self._sorted_overlapping_time(start_time1, start_time2, self.max_runtime_, end_time2)
            overlap += overlap_start + overlap_end

        # If there is no rollover, just calculate the overlap
        elif not rollover1 and not rollover2:
            overlap += self._sorted_overlapping_time(start_time1, start_time2, end_time1, end_time2)

        return overlap

    def get_task_starts(self, x: numpy.array, t: task) -> list[int]:
        """
        Returns ORDERED set of all starts inside the schedules, the list size is not fixed.
        """

        task_idx = self.tasks_.index(t)
        slice_start_idx = task_idx * self.max_exec_number_

        start_slice = x[slice_start_idx : slice_start_idx + self.max_exec_number_ : 1]
        starts = [round(start) for start in start_slice if round(start) != UNUSED_START]
        starts.sort()

        return starts

    def get_task_units(self, x: numpy.array, t: task) -> list[int]:
        """
        Returns units for the first, second, ... starting point in ascending order.
        """

        task_idx = self.tasks_.index(t)
        offset = len(self.tasks_) * self.max_exec_number_
        slice_start_idx = offset + task_idx * self.max_exec_number_

        task_starts = self.get_task_starts(x,t)

        unit_slice = x[slice_start_idx : max(slice_start_idx + self.max_exec_number_,len(task_starts)) : 1]
        units = [round(unit) for unit in unit_slice]

        if units == []:
            print("empty")

        return units

    def get_task_start_n(self, x: numpy.array, t: task, i: int) -> int:
        """
        Returns the ith starting point of task t.
        """
        return self.get_task_starts(x, t)[i]

    def get_task_unit_n(self, x: numpy.array, t: task, i: int) -> int:
        """
        Returns the unit of the ith starting point of task t.
        """
        units = self.get_task_units(x, t)
        return units[i]

    def get_task_exec_overlap(self, x: numpy.array, t1: task, t2: task) -> int:
        """
        Calculates the total overlap of all windows of task t1 and task t2.
        Note: There might be a rollover at the end of the scheduling period into the next one. This needs to be checked and managed accordingly.

        Arguments:
            x : numpy.array - current optimization value
            t1 : task - first task
            t2 : task - second task

        Returns:
            int, Number of overlaping time slots.
        """

        if t1 == t2:
            return 0

        overlap = 0  # Initialize overlap

        # Start time of tasks
        starts_task1 = self.get_task_starts(x, t1)
        starts_task2 = self.get_task_starts(x, t2)

        for idx1 in range(len(starts_task1)):
            start_time1 = starts_task1[idx1]
            end_time1 = (start_time1 + t1.get_exec_time()) % self.max_runtime_

            for idx2 in range(len(starts_task2)):
                # If the windows are not on the same unit, skip this combo
                unit1 = self.get_task_unit_n(x, t1, idx1)
                unit2 = self.get_task_unit_n(x, t2, idx2) 
                if unit1 != unit2:
                    continue

                start_time2 = starts_task2[idx2]
                end_time2 = (start_time2 + t2.get_exec_time()) % self.max_runtime_

                overlap += self._overlapping_time(start_time1, start_time2, end_time1, end_time2)

        return overlap

    def get_access_window_overlap(self, x: numpy.array, t1: task, t2: task) -> int:
        """
        Check for every access window of tasks t1 and t1, if they overlap.
        Note: There might be a rollover at the end of the scheduling period into the next one. This needs to be checked and managed accordingly.

        Arguments:
            x : numpy.array - current optimization value
            t1 : task - first task
            t2 : task - second task

        Returns:
            int, Number of overlaping time slots for every access window.
        """
        if t1 == t2:
            return 0

        # Check if access windows are present
        if t1.get_access_windows() is None or t2.get_access_windows() is None:
            return 0

        # Check for same access windows in dict
        windows_in_both = list(set(t1.get_access_windows().keys()).intersection(set(t2.get_access_windows().keys())))
        if len(windows_in_both) == 0:
            return 0

        # If we got here, there are access windows in windows_in_both present

        # Initialize overlap
        overlap = 0

        # Start time of tasks
        starts_task1 = self.get_task_starts(x, t1)
        starts_task2 = self.get_task_starts(x, t2)

        # Iterate over every index in both tasks
        for acc_idx in windows_in_both:

            # Save access window lists
            acc_windows1 = t1.get_access_windows()[acc_idx]
            acc_windows2 = t2.get_access_windows()[acc_idx]

            # Iterate over first task start points
            for idx1 in range(len(starts_task1)):
                start_time_task1 = starts_task1[idx1]

                # Iterate over second task start points
                for idx2 in range(len(starts_task2)):
                    start_time_task2 = starts_task2[idx2]

                    # Iterate over first window start and end
                    for rel_window_start1, rel_window_end1 in acc_windows1:
                        abs_start1 = (start_time_task1 + rel_window_start1) % self.max_runtime_
                        abs_end1 = (start_time_task1 + rel_window_end1) % self.max_runtime_

                        # Iterate over second window start and end
                        for rel_window_start2, rel_window_end2 in acc_windows2:
                            abs_start2 = (start_time_task2 + rel_window_start2) % self.max_runtime_
                            abs_end2 = (start_time_task2 + rel_window_end2) % self.max_runtime_

                            # Calculate overlapping time slots. Rollover gets taken car of by function
                            overlap += self._overlapping_time(abs_start1, abs_start2, abs_end1, abs_end2)

        return overlap

    def get_security_window_overlap(self, x: numpy.array, t1: task, t2: task) -> int:
        """
        Calculate overlaping security windows between two tasks. If both tasks do not have access windows, return 0.

        Arguments:
            x : numpy.array - current optimization value
            t1 : task - first task
            t2 : task - second task

        Returns:
            int, Number of overlaping time slots for exclusive windows.
        """
        # If both are identical
        if t1 == t2:
            return 0

        # If both tasks do not contain security windows
        if t1.get_exclusive_windows() is None and t2.get_exclusive_windows() is None:
            return 0

        # Initialize overlap
        overlap = 0

        # Start time of tasks
        starts_task1 = self.get_task_starts(x, t1)
        starts_task2 = self.get_task_starts(x, t2)

        # If t1 does not contain exclusive windows, skip
        if t1.get_exclusive_windows() is not None:
            for idx1 in range(len(starts_task1)):
                start_time1 = starts_task1[idx1]

                for idx2 in range(len(starts_task2)):
                    start_time2 = starts_task2[idx2]
                    end_time2 = start_time2 + t2.get_exec_time()

                    # Iterate over every security window pair
                    for sec_window_start1, sec_window_end1 in t1.get_exclusive_windows():
                        abs_start1 = start_time1 + sec_window_start1
                        abs_end1 = start_time1 + sec_window_end1

                        overlap += self._overlapping_time(abs_start1, start_time2, abs_end1, end_time2)

        # If t1 does not contain exclusive windows, skip
        if t2.get_exclusive_windows() is not None:
            for idx2 in range(len(starts_task2)):
                start_time2 = starts_task2[idx2]

                for idx1 in range(len(starts_task1)):
                    start_time1 = starts_task1[idx1]
                    end_time1 = start_time1 + t1.get_exec_time()

                    # Iterate over every security window pair
                    for sec_window_start2, sec_window_end2 in t2.get_exclusive_windows():
                        abs_start2 = start_time2 + sec_window_start2
                        abs_end2 = start_time2 + sec_window_end2

                        overlap += self._overlapping_time(abs_start2, start_time1, abs_end2, end_time1)

        return overlap

    def get_deadline_misses(self, x: numpy.array, t: task):
        """
        Returns a metric for missed deadlines between two executions.

        Arguments:
            x : numpy.array - current optimization value
            t1 : task - first task

        Returns:
            int - Number of time slots violating the deadline between tasks.
        """

        # If the task is not a critical task, no deadline is provided and the function returns with 0
        if not t.is_critical():
            return 0

        # Initialize penalty
        penalty = 0

        task_starts = self.get_task_starts(x, t)

        for idx in range(len(task_starts)):
            deadline_gap = 0

            # If last point is reached, keep wraping in mind
            if idx == len(task_starts) - 1:
                deadline_gap += self.max_runtime_ - task_starts[idx]
                deadline_gap += task_starts[0]

            # If last point is not reached, the gap is just the gap between both points
            else:
                deadline_gap += task_starts[idx + 1] - task_starts[idx]

            penalty += max(0, deadline_gap - t.get_periodic_deadline())
        
        return penalty

    def get_number_of_executions(self, x: numpy.array, t: task) -> int:
        """
        Returns the number of executions per scheduling table

        Arguments:
            x : numpy.array - current optimization value
            t1 : task - first task

        Returns:
            int - Number of executions.
        """

        start_times = self.get_task_starts(x, t)
        return len(start_times)
    
    def get_diference_from_optimum_executions(self, x: numpy.array, t: task) -> int:
        """
        Returns the difference from the optimal number of executions for every task in conjunction with the maximum amount of executions and the periodic deadlines.
        Has a higher penalty for 0 executions.

        Arguments:
            x : numpy.array - current optimization vector
            t: task - the task

        Returns:
            The difference between the optimum and the current execution value.
        """
        exec_amount = self.get_number_of_executions(x,t)
        if exec_amount == 0:
            return NO_EXEC_SEVERITY
        
        expected_executions = math.ceil(self.max_runtime_ / t.get_periodic_deadline())
        difference = abs(exec_amount - expected_executions)
        return difference

    def __call__(self, x: numpy.array, *args: Any) -> Any:
        """
        Many entry to cost calculation of optimizers. Takes in x as numpy array. Other arguments are not used.

        Arguments:
            x : numpy.array - vector that is optimized by optimization problem

        Returns:
            Any - the cost of the schedule
        """
        penalty = 0

        for t1 in range(len(self.tasks_)):
            for t2 in range(t1, len(self.tasks_)):

                # Check if task execution overlap
                penalty += TASK_OVERLAP_PENALTY * self.get_task_exec_overlap(x, self.tasks_[t1], self.tasks_[t2])

                # Check if task access windows overlap
                penalty += ACCESS_WINDOW_PENALTY * self.get_access_window_overlap(
                    x, self.tasks_[t1], self.tasks_[t2]
                )

                # Check if task security windows overlap
                penalty += SEC_WINDOW_PENALTY * self.get_security_window_overlap(
                    x, self.tasks_[t1], self.tasks_[t2]
                )

            # For multiple execution windows inside one TDMA schedule table
            penalty += DEADLINE_MISS_PENALTY * self.get_deadline_misses(x, self.tasks_[t1])

            # Add a small cost for more execuitons per schedule table
            penalty += MULTIPLE_EXEC_COST * self.get_diference_from_optimum_executions(x, self.tasks_[t1])

        return penalty


class annealing_schedule_creator(polyhedron_heuristics.schedule_tools.abstract_schedule_creator):
    """
    Scheduling creator based on simulated / dual annealing. The most work is done in the cost function class.
    Further work might start from there.

    Members:
        cost_function_ : soft_cost_function - The used cost function
    """

    def __init__(self, cost_function: soft_cost_function) -> None:
        self.cost_function = cost_function
        super().__init__()

    def callback(self, x, tasks : list[task], e=None, context=None):
        """
        Callback function to call from the optimizer. Only prints the result and triggers exit, if 0 is reached.
        """
        print(str(x) + " with cost value: " + str(e))
        for t in tasks:
            all_task_starts = self.cost_function.get_task_starts(x,t)
            for current_start in all_task_starts:
                current_unit = self.cost_function.get_task_unit_n(x,t,all_task_starts.index(current_start))

                print("Task : " + str(t.get_id()) + " Start: " + str((current_start) % self.cost_function.max_runtime_) + " End: " + str((current_start + t.get_exec_time()) % self.cost_function.max_runtime_) + " Unit: " + str(current_unit))
                if t.get_access_windows() is not None:
                    windows = t.get_access_windows()
                    for access_window_key in windows.keys():
                        for acces_window in windows[access_window_key]:
                            print("Access window position: " + str((current_start + acces_window[0]) % self.cost_function.max_runtime_) + " until: " + str((current_start + acces_window[1]) % self.cost_function.max_runtime_))
                        
                    
        
        if e == 0:
            return True
        return False

    def convert_output_to_schedule(
        self, x: numpy.array, no_units: int, all_tasks: list[task]
    ) -> polyhedron_heuristics.schedule_tools.abstract_schedule:
        """
        Convert a given result to an abstract_schedule class.

        Arguments:
            x : numpy.array - The optimizer result
            no_units : int - The number of used execution units
            all_tasks : list[task] - A list of all used tasks

        Returns:
            abstract_schedule : Generated abstract_schedule object
        """

        task_amount = len(all_tasks)

        # Get amount of shared resources
        unique_share_resources: set[int] = set()
        for t in all_tasks:
            if t.get_access_windows() is not None:
                unique_share_resources.update(t.get_access_windows().keys())
        number_shared_resources = len(unique_share_resources)

        # Create empty schedule
        current_schedule = polyhedron_heuristics.schedule_tools.abstract_schedule(
            all_tasks[0].get_resolution(),
            no_units,
            number_shared_resources,    # Processor is handled in constructor
            periodic_deadline=self.cost_function.max_runtime_,
        )

        # Cycle through every task
        for i in range(task_amount):
            current_task = all_tasks[i]
            starts = self.cost_function.get_task_starts(x, current_task)
            units = self.cost_function.get_task_units(x, current_task)

            # Cycle through every starting index of that task
            for current_start_idx in range(len(starts)):
                current_start = starts[current_start_idx]
                unit = units[current_start_idx]

                # Create the next polyhedron and enter it into the schedule
                current_polyhedron = platform_exports.task_tools.polyhedron(
                    no_units,
                    unit,
                    current_task.get_exec_time(),
                    i + 1,
                    current_task.get_access_windows(),
                    current_task.get_exclusive_windows(),
                    number_shared_resources
                )
                current_schedule.add_polyhedron_to_schedule(current_polyhedron, current_start)

        return current_schedule

    def create_schedule(
        self,
        sched_tasks: list[task],
        no_units: int,
    ) -> polyhedron_heuristics.schedule_tools.abstract_schedule:

        # Set same timing resolution for all tasks
        timing_res = platform_exports.task_tools.get_timing_resolution(sched_tasks, platform_minimum=PLATFORM_MINIMUM_RES)
        for t in sched_tasks:
            t.fit_to_resolution(timing_res)

        # Get the max number of executions from the cost function.
        max_exec_number = self.cost_function.max_exec_number_

        N_D = len(sched_tasks)

        # x is build as [t_start00, t_start_01, ... , t_startN_D0, t_startN_D1, ... , u_00, u_01, ...]
        x = numpy.zeros(2 * max_exec_number * N_D)
        lb = numpy.full(x.shape, 0)  # set lower bound for all to 0
        lb[: N_D * max_exec_number :] = UNUSED_START  # set lower bound for starts to -1

        ub = numpy.full(x.shape, self.cost_function.max_runtime_)  # set upper bound for tasks to schedule table length
        ub[N_D * max_exec_number : :] = no_units - 1  # set upper bound for units to no_units

        bounds = scipy.optimize.Bounds(lb, ub)

        #  Adapt initial temperature to number of tasks
        # init_temp = 5e4 * (1 - 1 / no_units * no_units)

        # Do the annealing process
        res = scipy.optimize.dual_annealing(
            self.cost_function, bounds, maxiter=MAX_ITERATIONS, callback=lambda x,e,context : self.callback(x,sched_tasks,e,context), x0=x
        )

        # Print the resulting stats
        print(str(res["x"]) + " with cost value: " + str(res["fun"]))

        # Convert the annealing result to a schedule
        schedule = self.convert_output_to_schedule(res["x"], no_units, sched_tasks)
        return schedule
