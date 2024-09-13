import math
import platform_exports.platform_tools
import platform_exports.task_tools
import platform_exports.unit_tools
import numpy
import polyhedron_heuristics.schedule_tools as schedule_tools

MAX_RETRY_ITERATIONS = 5


class block_step(schedule_tools.abstract_schedule_creator):
    """
    Algorithm to create a schedule by creating polyhedrons and formulating this as a geometric programming problem.
    This class extends the abstract schedule creator and provides the create_schedule function.

    This class has no members and only provides the implementation of the interface.
    """

    def __init__(self, deadline : int = math.inf) -> None:
        self.deadline_ = deadline

    def has_parallel_execution(self, poly: platform_exports.task_tools.polyhedron) -> bool:
        """
        Checks if a polyhedron has a parallel execution on another unit, e.g. a security window.

        Arguments:
            poly : polyhedron - the polyhedron to check.

        Returns:
            bool - True if parallel execution is present.
        """
        scheduled_on_unit: list[bool] = []
        for u in range(poly.tensor_.shape[schedule_tools.u_axis]):
            scheduled_on_unit.append(numpy.sum(poly.tensor_[0, u, :]) != 0)

        return sum(bool(x) for x in scheduled_on_unit) != 1

    def get_schedule_cost(
        self, schedule: schedule_tools.abstract_schedule, polyhedron: platform_exports.task_tools.polyhedron
    ) -> tuple[int, int]:
        """
        Get the cost of a given schedule.
        If the polyhedron has a parallel execution, the time slots blocked before the parallel execution are considered blocked.
        If the polyhedron has no parallel execution, the overshoot over the latest executed task is considered it's cost.

        Arguments:
            schedule : abstract_schedule - The existing schedule
            polyhedron : polyhedron - The polyhedron that is placed

        Return:
            tuple[int,int] - Tuple of the cost (0) and the time (1)
        """

        # Get the earliest time, a task can be executed
        t_temp = schedule.get_first_free_t(polyhedron)

        # If the task can only be executed at the end i.e. the test ran through every position, return with inf cost and time
        if t_temp == schedule.S_matrix_.shape[schedule_tools.t_axis]:
            return math.inf, math.inf

        # Register the polyhedron to the matrix and get the latest free space
        test_matrix = schedule_tools.register_polyhedron(schedule, t_temp, polyhedron)
        t_max_after = schedule_tools.get_last_first_empty(test_matrix)

        # If the polyhedron has parallel execution, calculate the cost as the number of zero elements (length - nonzero)
        if self.has_parallel_execution(polyhedron):
            cost = 0
            for u in range(0, test_matrix.shape[schedule_tools.u_axis]):
                test_slice = test_matrix[0, u, 0 : t_max_after[u]]
                cost += t_max_after[u] - numpy.count_nonzero(test_slice)

        # If the polyhedron has no parallel execution, calculate the cost as the number of overshooting elements
        # (3 * max(t_max_after) - nonzeros)
        else:
            nonzeros = 0
            for u in range(0, test_matrix.shape[schedule_tools.u_axis]):
                test_slice = test_matrix[0, u, 0 : max(t_max_after)]
                nonzeros += numpy.count_nonzero(test_slice)

            cost = 3 * max(t_max_after) - nonzeros

        return cost, t_temp

    def place_next_polyhedron(
        self,
        schedule: schedule_tools.abstract_schedule,
        polyhedrons: list[platform_exports.task_tools.polyhedron],
        tasks: list[platform_exports.task_tools.task],
    ) -> platform_exports.task_tools.polyhedron:
        """
        Place the next polyhedron onto the schedule by picking the starting time as the minumum of the cost function.

        Arguments:
            schedule: schedule_tools.abstract_schedule - The schedule to place the elements onto
            polyhedrons: list[polyhedron] - The list of polyhedrons needing placement
            tasks: list[task] - A task list for indexing porposes.

        Returns:
            polyhedron - The placed polyhedron.
        """
        # Initialize cost and starting time
        cost: int = math.inf
        t_used = 0

        for polyh in polyhedrons:
            # Calculate cost and starting time
            current_cost, t_tmp = self.get_schedule_cost(schedule, polyh)

            # Find minumum cost
            if current_cost < cost:
                cost = current_cost
                used_polyhedron = polyh
                t_used = t_tmp

            # If cost is identical but task is considered critical, choose this task instead
            elif current_cost == cost and tasks[polyh.get_task_id() - 1].is_critical():
                used_polyhedron = polyh
                t_used = t_tmp

        # Exeception if no polyhedron can be placed.
        if cost == math.inf:
            raise RuntimeError("Infinite cost for all polyhedrons!")

        # If polyhedron is found, add it to schedule and return
        schedule.add_polyhedron_to_schedule(used_polyhedron, t_used)
        return used_polyhedron

    def get_associated_task(
        self,
        polyh: platform_exports.task_tools.polyhedron,
        task_dict: dict[platform_exports.task_tools.task, list[platform_exports.task_tools.polyhedron]],
    ) -> platform_exports.task_tools.task:
        """
        Get the assiciated task for a polyhedron.

        Arguments:
            polyh : polyhedron - The polyhedron to check.
            task_dict: dict[task, list[polyhedron]] - The task-to-polyhedron dictionary

        Returns:
            task - The associated task
        """
        return [associated_task for associated_task in task_dict.keys() if polyh in task_dict[associated_task]][0]
    
    def get_number_of_shared_res(self, task_list : list[platform_exports.task_tools.task]) -> int:
        """
        Calculates the number of unique shared resource indexes in the task access.
        
        Arguments:
            task_list : list[task] - The list of all tasks

        Returns:
            int - The number of unique shared resources
        """
        shared_res : set[int]= set()
        for t in task_list:
            if t.get_access_windows() is not None:
                shared_res.update(t.get_access_windows().keys())
        return len(shared_res)
    
    def get_task_from_id(self, id : int, all_tasks : list[platform_exports.task_tools.task]) -> platform_exports.task_tools.task:
        """
        Returns the task for a given ID.

        Arguments:
            id : int - The ID.
            all_tasks : list[task] - A list of all tasks to choose from

        Returns:
            task - The corresponding task
        """
        tasks_with_id = [id_task for id_task in all_tasks if id_task.get_id() == id]
        if len(tasks_with_id) > 1:
            raise RuntimeError("Found more than 1 task with id " + str(id))
        elif len(tasks_with_id) == 0:
            raise RuntimeError("No task with that ID found")
    
        return tasks_with_id[0]

    def iterative_improve(
        self, schedule: schedule_tools.abstract_schedule, all_tasks: list[platform_exports.task_tools.task], units: list[platform_exports.unit_tools.unit]
    ) -> tuple[schedule_tools.abstract_schedule, int]:
        """
        Iterative improvement for given abstract schedule.

        Arguments:
            schedule : abstract_schedule - The schedule to be improved.
            all_tasks : list[task] - List of all tasks (scheduled and unscheduled)

        Returns:
            tuple[abstract_schedule, int] - Tuple of schedule and number of non scheduled tasks
        """

        # Get violating task
        no_tasks = int(numpy.unique(schedule.S_matrix_).size)
        
        latest_unit = schedule.get_latest_unit()
        latest = schedule.get_latest_index_on_unit(latest_unit)
        latest_task_idx = schedule.get_task_at(latest_unit, latest)

        # Clear schedule of violating task
        schedule.clear_task(latest_task_idx)

        # Get best fitting swapping
        violating_polyhedrons = platform_exports.task_tools.create_task_polyhedrons(
            self.get_task_from_id(latest_task_idx, all_tasks), self.get_number_of_shared_res(all_tasks), units
        )
        min_cost = math.inf
        swapped_task_idx: list[int] = []

        for u in range(schedule.get_no_units()):
            current_cost, swap_task_idxs = schedule.get_swapping_tasks(violating_polyhedrons[u], all_tasks)
            if current_cost < min_cost:
                min_cost = current_cost
                swapped_task_idx = swap_task_idxs

        # Remove the tasks to clear from the schedule
        for task_to_clear in swapped_task_idx:
            schedule.clear_task(task_to_clear)

        # Register violating task back to the schedule
        self.place_next_polyhedron(schedule, violating_polyhedrons, all_tasks)

        # Get not scheduled tasks
        non_scheduled_tasks: list[platform_exports.task_tools.task] = [
            d for d in all_tasks if d.get_id() not in schedule.get_scheduled_tasks()
        ]

        # Create polyhedron dict for non-scheduled tasks
        polyhedron_dict: dict[platform_exports.task_tools.task, list[platform_exports.task_tools.polyhedron]] = {
            current_task: platform_exports.task_tools.create_task_polyhedrons(current_task, self.get_number_of_shared_res(all_tasks), units)
            for current_task in non_scheduled_tasks
        }

        # Try to place all tasks
        while non_scheduled_tasks:
            all_polyhedrons : list[platform_exports.task_tools.polyhedron] = []
            for key_task in non_scheduled_tasks:
                all_polyhedrons.extend(polyhedron_dict[key_task])
            try:
                placed_polyhedron = self.place_next_polyhedron(schedule, all_polyhedrons, all_tasks)
            except RuntimeError as e:
                print("Could still not place a polyhedron")
                break

            placed_task = self.get_associated_task(placed_polyhedron, polyhedron_dict)

            # Remove tasks from the lists.
            non_scheduled_tasks.remove(placed_task)

        # Count how many tasks are not placed
        no_tasks_end = int(numpy.unique(schedule.S_matrix_).size)
        print("Lost tasks: " + str(no_tasks - no_tasks_end))

        return schedule, no_tasks - no_tasks_end

    def create_schedule(
        self,
        sched_tasks: list[platform_exports.task_tools.task],
        units: list[platform_exports.unit_tools.unit]
    ) -> schedule_tools.abstract_schedule:
        """
        Main entry point for schedule creation based on the block step algorithm.
        (HINT: It is relatively fast, but not effective for workloads with high processor utilisation.)

        Arguments:
            sched_tasks: list[task] - Tasks to schedule
            no_units: int - Number of homogeneous execution units

        Returns:
            abstract_schedule - An abstract schedule representation
        """

        # Make sure, no task has ID 0, that is needed for spacing
        for check_task in sched_tasks:
            if check_task.get_id() == 0:
                raise RuntimeError("0 not allowed as task ID.")

        # Fit the tasks to the resolution (no effect if already done)
        timing_res = platform_exports.task_tools.get_timing_resolution(sched_tasks)
        for t in sched_tasks:
            t.fit_to_resolution(timing_res)

        # Get number of shared resources
        no_shared_res = self.get_number_of_shared_res(sched_tasks)

        # Get number of units
        no_units = len(units)
        
        # Initialise schedule with task resolution and number of units
        current_schedule = schedule_tools.abstract_schedule(timing_res, no_units,no_shared_resources=no_shared_res,periodic_deadline=self.deadline_)

        # Create a dictionary mapping tasks to polyhedrons
        polyhedron_dict: dict[platform_exports.task_tools.task, list[platform_exports.task_tools.polyhedron]] = {
            current_task: platform_exports.task_tools.create_task_polyhedrons(current_task, no_shared_res, units)
            for current_task in sched_tasks
        }

        # Create dictionary of already scheduled tasks
        marked_scheduled: dict[platform_exports.task_tools.task, bool] = {
            current_task: False for current_task in sched_tasks
        }

        # Run algorithm for as long as there is still a task to be scheduled
        while False in marked_scheduled.values():

            # Collect the polyhedrons of all not yet scheduled tasks (False entry for the task in marked_scheduled)
            considered_polyhedrons: list[platform_exports.task_tools.polyhedron] = []
            for current_task in sched_tasks:
                if marked_scheduled[current_task] == False:
                    considered_polyhedrons.extend(polyhedron_dict[current_task])

            # Try to place the polyhedron. If it does not work, raise an error and break with the half finished schedule.
            try:
                next_polyhedron = self.place_next_polyhedron(current_schedule, considered_polyhedrons, sched_tasks)
            except RuntimeError as e:
                print(e)
                break

            # Get the placed task and mark it as scheduled
            chosen_task = self.get_associated_task(next_polyhedron, polyhedron_dict)
            marked_scheduled[chosen_task] = True

        # If there are still tasks not scheduled, do iterative improvement
        iterations = 0
        while False in marked_scheduled.values() and iterations < MAX_RETRY_ITERATIONS:
            current_schedule, _ = self.iterative_improve(current_schedule, sched_tasks, units)

            # Get task IDs of scheduled tasks
            currently_scheduled_tasks = current_schedule.get_scheduled_tasks()
            for task_it in sched_tasks:
                marked_scheduled[task_it] = task_it in currently_scheduled_tasks

            iterations += 1

        if False not in marked_scheduled.values():
            current_schedule.set_valid()

        # Return the schedule
        return current_schedule
