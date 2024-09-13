import math
from typing import Any
import numpy
import scipy.optimize
import platform_exports
import platform_exports.platform_tools
from platform_exports.task_tools import task
import platform_exports.task_tools
import platform_exports.unit_tools
import polyhedron_heuristics.schedule_tools
import scipy

# Definitions
TASK_OVERLAP_PENALTY = 3000
ACCESS_WINDOW_PENALTY = 500
RUNTIME_OVERSHOOT_PENALTY = 2000
DEADLINE_MISS_PENALTY = 1500
MULTIPLE_EXEC_COST = 250
TASK_OVERRUN_PENALTY = 15000
NO_EXEC_SEVERITY = 100
UNUSED_START = -1
MAX_ITERATIONS = 2000
PLATFORM_MINIMUM_RES = 100

class heterogeneous_cost_function:
    """
    Cost function class that gets called to determine the penaltys and costs of a proposed schedule.

    Members:
        tasks_: list[task] - A list of all tasks used for indexing
        unit_lists_: int - The list of (heterogeneous) execution units
        max_runtime_: int = math.inf - The maximum allowed runtime
        max_exex_number_: int = 1 - The number of allowed executions per task in this TDMA schedule table

    Definition of functions is based on optimizing the vector x : numpy.array, which is build like this:
    [t_00, t_00, ..., t_10, t_11, ..., u_00, u_01, ..., u_10, u_11, ...]
    At first, the time of every execution is presented.
    Afterward, execution unit for each starting point is presented.

    If a start is not used, this is determined by the UNUSED_START definition.
    """

    def __init__(
        self,
        tasks: list[task],
        unit_list : list[platform_exports.unit_tools.unit],
        max_runtime: int = math.inf,
        max_exec_number: int = 1,
    ) -> None:
        self.tasks_ = tasks
        self.units_ = unit_list
        self.max_runtime_ = max_runtime
        self.max_exec_number_ = max_exec_number
        self.task_unit_dict : dict[task,dict[int,int]]= {}

    def get_real_unit_from_task_unit(self, t: task, u_index:int) -> int:
        """
        This function translates the local task unit to the "real" global unit. This is necessary because of the constraints placed on the tasks.
        """
        return self.task_unit_dict[t][u_index]

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

    def _get_task_units_indeces(self, x: numpy.array, t: task) -> list[int]:
        """
        Returns units for the first, second, ... starting point in ascending order.
        """
        task_idx = self.tasks_.index(t)
        offset = len(self.tasks_) * self.max_exec_number_
        slice_start_idx = offset + task_idx * self.max_exec_number_

        task_starts = self.get_task_starts(x, t)

        unit_slice = x[slice_start_idx : max(slice_start_idx + self.max_exec_number_, len(task_starts)) : 1]
        units = [round(unit) for unit in unit_slice]

        if units == []:
            print("empty")

        return units

    def get_task_start_n(self, x: numpy.array, t: task, i: int) -> int:
        """
        Returns the ith starting point of task t.
        """
        return self.get_task_starts(x, t)[i]

    def get_task_unit_n(self, x: numpy.array, t: task, i: int) -> platform_exports.unit_tools.unit:
        """
        Returns the GLOBAL unit of the ith starting point of task t.
        """
        units = self._get_task_units_indeces(x, t)

        # Important! Get the GLOBAL unit
        unit_index = self.task_unit_dict[t][units[i]]
        return self.units_[unit_index]
    
    def get_unit_type(self, local_unit_index : int) -> int:
        """
        Returns the unit type of the unit index.
        """
        return self.units_[local_unit_index].get_type()    

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
        overlap = 0  # Initialize overlap

        # Start time of tasks
        starts_task1 = self.get_task_starts(x, t1)
        starts_task2 = self.get_task_starts(x, t2)

        for idx1 in range(len(starts_task1)):
            start_time1 = starts_task1[idx1]
            unit1 = self.get_task_unit_n(x, t1, idx1)
            end_time1 = (start_time1 + t1.get_exec_time(unit1.get_type())) % self.max_runtime_

            for idx2 in range(len(starts_task2)):
                # Only skip, if the task and the execution start is identical
                if t1 == t2 and idx1 == idx2:
                    continue

                # If the windows are not on the same unit, skip this combo
                unit2 = self.get_task_unit_n(x, t2, idx2)

                if unit1 != unit2:
                    continue

                start_time2 = starts_task2[idx2]
                end_time2 = (start_time2 + t2.get_exec_time(unit2.get_type())) % self.max_runtime_

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

        # Initialize overlap
        overlap = 0

        # Start time of tasks
        task_starts1 = self.get_task_starts(x, t1)
        task_starts2 = self.get_task_starts(x, t2)

        # Iterate over first task start points
        for idx1 in range(len(task_starts1)):
            unit1 = self.get_task_unit_n(x,t1,idx1)
            acc_windows1 = t1.get_access_windows()[unit1.get_type()]
            start_time_task1 = task_starts1[idx1]

            # Iterate over second task start points
            for idx2 in range(len(task_starts2)):
                unit2 = self.get_task_unit_n(x,t2,idx2)
                acc_windows2 = t2.get_access_windows()[unit2.get_type()]
                start_time_task2 = task_starts2[idx2]

                # Iterate over first window
                for acc_window1 in acc_windows1:
                    abs_start1 = (start_time_task1 + acc_window1.get_start()) % self.max_runtime_
                    abs_end1 = (start_time_task1 + acc_window1.get_stop()) % self.max_runtime_

                    # Iterate over second window
                    for acc_window2 in acc_windows2:

                        if acc_window1.get_resource() != acc_window2.get_resource():
                            continue

                        abs_start2 = (start_time_task2 + acc_window2.get_start()) % self.max_runtime_
                        abs_end2 = (start_time_task2 + acc_window2.get_stop()) % self.max_runtime_

                        # Calculate overlapping time slots. Rollover gets taken car of by function
                        overlap += self._overlapping_time(abs_start1, abs_start2, abs_end1, abs_end2)

        return overlap

    def get_deadline_misses(self, x: numpy.array, t: task) -> int:
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
        exec_amount = self.get_number_of_executions(x, t)
        if exec_amount == 0:
            return NO_EXEC_SEVERITY

        if t.get_periodic_deadline() < 1:
            expected_executions = 1
        else:
            expected_executions = math.ceil(self.max_runtime_ / t.get_periodic_deadline())

        difference = abs(exec_amount - expected_executions)
        return difference
    
    def get_task_overrun(self, x: numpy.array, t:task) -> int:
        """
        Returns the difference from the optimal number of executions for every task in conjunction with the maximum amount of executions and the periodic deadlines.
        Has a higher penalty for 0 executions.

        Arguments:
            x : numpy.array - current optimization vector
            t: task - the task

        Returns:
            The difference between the optimum and the current execution value.
        """
        penalty = 0
        for idx in range(len(self.get_task_starts(x,t))):
            current_unit = self.get_task_unit_n(x,t,idx)

            penalty += max(0, t.get_exec_time(current_unit.get_type()) - self.max_runtime_)

        return penalty

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
                penalty += ACCESS_WINDOW_PENALTY * self.get_access_window_overlap(x, self.tasks_[t1], self.tasks_[t2])

            # Check if the task execution on the current unit is longer than the scheduling time
            penalty += TASK_OVERRUN_PENALTY * self.get_task_overrun(x, self.tasks_[t1])

            # Check, if the deadline is reached, done for multiple execution windows inside one TDMA schedule table
            penalty += DEADLINE_MISS_PENALTY * self.get_deadline_misses(x, self.tasks_[t1])

            # Add a small cost for more execuitons per schedule table
            penalty += MULTIPLE_EXEC_COST * self.get_diference_from_optimum_executions(x, self.tasks_[t1])

        return penalty

class annealing_schedule_creator(polyhedron_heuristics.schedule_tools.abstract_schedule_creator):
    """
    Scheduling creator based on simulated / dual annealing. The most work is done in the cost function class.

    Members:
        cost_function_ : soft_cost_function - The used cost function
    """

    def __init__(self, cost_function: heterogeneous_cost_function) -> None:
        self.cost_function = cost_function
        super().__init__()

    def callback(self, x, tasks: list[task], e=None, context=None):
        """
        Callback function to call from the optimizer. Only prints the result and triggers exit, if 0 is reached.
        """
        print(str(x) + " with cost value: " + str(e))
        for t in tasks:
            all_task_starts = self.cost_function.get_task_starts(x, t)
            for current_start in all_task_starts:
                current_unit = self.cost_function.get_task_unit_n(x, t, all_task_starts.index(current_start))

                print(
                    "Task : "
                    + str(t.get_id())
                    + " Start: "
                    + str((current_start) % self.cost_function.max_runtime_)
                    + " End: "
                    + str((current_start + t.get_exec_time(current_unit.get_type())) % self.cost_function.max_runtime_)
                    + " Unit: "
                    + str(self.cost_function.units_.index(current_unit))
                )

                if t.get_access_windows() is not None and t.get_access_windows()[current_unit.get_type()]:
                    windows = t.get_access_windows()[current_unit.get_type()]

                    for access_window in windows:
                        print(
                            "Access window position: "
                            + str((current_start + access_window.get_start()) % self.cost_function.max_runtime_)
                            + " until: "
                            + str((current_start + access_window.get_stop()) % self.cost_function.max_runtime_)
                        )

        if e == 0:
            return True
        
        return False

    def convert_output_to_schedule(
        self, x: numpy.array, units: list[platform_exports.unit_tools.unit], all_tasks: list[task]
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
        # Get amount of shared resources
        unique_share_resources: set[int] = set()
        for t in all_tasks:
            if t.get_access_windows() is not None:
                unique_share_resources.update(t.get_access_windows().keys())
        number_shared_resources = len(unique_share_resources)

        # Create empty schedule
        current_schedule = polyhedron_heuristics.schedule_tools.abstract_schedule(
            all_tasks[0].get_resolution(),
            len(units),
            number_shared_resources,  # Processor is handled in constructor
            periodic_deadline=self.cost_function.max_runtime_,
        )

        # Cycle through every task
        for current_task in all_tasks:
            starts = self.cost_function.get_task_starts(x, current_task)

            # Cycle through every starting index of that task
            for current_start_idx in range(len(starts)):
                current_start = starts[current_start_idx]
                unit = self.cost_function.get_task_unit_n(x,current_task,current_start_idx)

                used_access_windows = None if current_task.get_access_windows() is None else current_task.get_access_windows()[unit.get_type()]

                # Create the next polyhedron and enter it into the schedule
                current_polyhedron = platform_exports.task_tools.polyhedron(
                    len(units),
                    units.index(unit),
                    current_task.get_exec_time(unit.get_type()),
                    current_task.get_id(),
                    used_access_windows,
                    number_shared_resources,
                )
                current_schedule.add_polyhedron_to_schedule(current_polyhedron, current_start)

        return current_schedule
    
    def check_for_valid_exec_unit(self, x: numpy.array, all_tasks : list[task]) -> int :
        """
        Function to show, if the execution of tasks on the respective unit is valid
        """
        non_fitting = 0
        for task in all_tasks:
            units = self.cost_function._get_task_units_indeces(x, task)
            non_fitting += len([u for u in units if u not in task.get_supported_exec_unit_indeces()])

        return non_fitting
    

    def create_local_to_global_dict(self, tasks : list[task], units : list[platform_exports.unit_tools.unit]) -> dict[task,dict[int,int]]:
        """
        Function to generate a dictionary, translating the local continous indexing of a task to the global indexing of the schedule.

        Arguments:
            tasks : list[task] - A list of all tasks to be scheduled on this platform
            units : list[unit] - A list of all available units

        Returns:
            dict[task,dict[int,int]] - The translation between local and global units.
        """
        global_unit_index_dict : dict[platform_exports.unit_tools.unit,int] = {}
        index = 0

        for local_unit in units:
            global_unit_index_dict[local_unit] = index
            index += 1

        local_global_dict : dict[task,dict[int,int]] = {}

        for t in tasks:
            supported_unit_indeces = t.get_supported_exec_unit_indeces()
            executable_units = [u for u in units if u.get_type() in supported_unit_indeces]
            
            local_dict : dict[int,int]= {}
            local_index = 0
            for exec_unit in executable_units:
                local_dict[local_index] = global_unit_index_dict[exec_unit]
                local_index+=1

            local_global_dict[t] = local_dict

        return local_global_dict
    
    def create_upper_bounds(self, tasks : list[task], unit_dict : dict[task,dict[int,int]], exec_max : int) -> numpy.array:
        """
        Creates the upper bounds array for the task decision variables for execution. Because of the different execution units,
        different upper bounds are needed.

        Arguments:
            tasks : list[task] - The list of all tasks to be scheduled
            exec_max : int - The maximum amount a task can be executed

        Returns:
            numpy.array - The upper bounds
        """
        # Create the array with 0s
        ub_array = numpy.zeros(exec_max * len(tasks))
        
        index = 0
        for t in tasks:

            # Fill the range of the array with the maximum local key (the maximum number available)
            ub_array[index * exec_max:(index+1)*exec_max] = max(unit_dict[t].keys())
            
            # Increment index for next task
            index += 1
        
        return ub_array


    def create_schedule(
        self,
        sched_tasks: list[task],
        units: list[platform_exports.unit_tools.unit],
    ) -> polyhedron_heuristics.schedule_tools.abstract_schedule:

        # Set same timing resolution for all tasks
        timing_res = platform_exports.task_tools.get_timing_resolution(
            sched_tasks, platform_minimum=PLATFORM_MINIMUM_RES
        )
        for t in sched_tasks:
            t.fit_to_resolution(timing_res)

        # Get the max number of executions from the cost function.
        max_exec_number = self.cost_function.max_exec_number_

        self.cost_function.task_unit_dict = self.create_local_to_global_dict(sched_tasks, units)

        N_D = len(sched_tasks)

        # x is build as [t_start00, t_start_01, ... , t_startN_D0, t_startN_D1, ... , u_00, u_01, ...]
        x = numpy.zeros(2 * max_exec_number * N_D)
        lb = numpy.full(x.shape, 0)  # set lower bound for all to 0
        lb[: N_D * max_exec_number :] = UNUSED_START  # set lower bound for starts to -1

        ub = numpy.full(x.shape, self.cost_function.max_runtime_)  # set upper bound for tasks to schedule table length
        ub[N_D * max_exec_number : :] = self.create_upper_bounds(sched_tasks,self.cost_function.task_unit_dict,max_exec_number)  # set upper bound for units to no_units

        bounds = scipy.optimize.Bounds(lb, ub)

        #  Adapt initial temperature to number of tasks
        # init_temp = 5e4 * (1 - 1 / no_units * no_units)

        # Do the annealing process
        res = scipy.optimize.dual_annealing(self.cost_function,bounds,maxiter=MAX_ITERATIONS,callback=lambda x, e, context: self.callback(x, sched_tasks, e, context),x0=x,)

        # Print the resulting stats
        print(str(res["x"]) + " with cost value: " + str(res["fun"]))

        # Convert the annealing result to a schedule
        schedule = self.convert_output_to_schedule(res["x"], units, sched_tasks)
        return schedule
