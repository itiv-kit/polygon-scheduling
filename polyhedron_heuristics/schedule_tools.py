import io
import math
import sys
from platform_exports.task_tools import polyhedron, acc_axis, t_axis, u_axis
import platform_exports.unit_tools
import numpy

EMPTY_SYMBOL = None
MAX_DURATION_SETUP = 4000


class abstract_schedule:
    """
    Class to represent a schedule independant of the platform.

    Members:
        timing_resolution_: int - The used timing resolution for the schedule
        S_matrix_: numpy.ndarray - The internal representation of the schedule,
            where the first dimension represents shared ressources,
            the second dimension represents the execution units and
            the third dimension represents time.
        valid_ : bool - Flag to signify, if the schedule is created or still unfinished
    """

    def __init__(
        self,
        timing_resolution: int,
        no_units_: int,
        no_shared_resources: int = 1,
        S_matrix: numpy.ndarray | None = None,
        periodic_deadline: int | None = None,
    ) -> None:
        self.timing_resolution_ = timing_resolution

        # If no S_matrix is provided, generate stub
        if S_matrix is None:
            self.valid_ = False

            # If no periodic deadline is provided, generate the frame with a given max duration used for setup
            if periodic_deadline is None or periodic_deadline is math.inf:
                self.S_matrix_ = numpy.zeros(
                    shape=[no_shared_resources + 1, no_units_, MAX_DURATION_SETUP], dtype=numpy.int16
                )
            else:
                self.S_matrix_ = numpy.zeros(
                    shape=[no_shared_resources + 1, no_units_, periodic_deadline],
                    dtype=numpy.int16,
                )
        else:
            self.valid_ = True
            self.S_matrix_ = S_matrix

    def set_valid(self) -> None:
        """
        Mark the schedule as valid. Rais error, if it already is valid.
        """
        if self.valid_:
            raise RuntimeError("Already valid!")
        self.valid_ = True

    def get_timing_resolution(self) -> int:
        """
        Return the timing resolution of the schedule.
        """
        return self.timing_resolution_

    def get_length(self) -> int:
        """
        Return the timing length of the S_matrix
        """
        return self.S_matrix_.shape[t_axis]

    def get_no_units(self) -> int:
        """
        Return the number of execution units.
        """
        return self.S_matrix_.shape[u_axis]

    def get_scheduled_tasks(self) -> list[int]:
        """
        Return a list of all scheduled tasks.
        """
        return numpy.unique(self.S_matrix_[self.S_matrix_ != 0])
    
    def get_shared_resources(self) -> int:
        """
        Return the number of shared resources.
        """
        return self.S_matrix_.shape[acc_axis]

    def get_first_free_t(self, polyhedron: polyhedron) -> int:
        """
        Get the first free time slot for a given polyhedron.
        Returns the length of the matrix, if no suitable place is found.

        Arguments:
            polyhedron: polyhedron - The polyhedron that is used to search the free time for.

        """
        t_x = polyhedron.tensor_.shape[t_axis]
        length_diff = self.S_matrix_.shape[t_axis] - t_x
        for t in range(0, length_diff):
            used_submatrix = self.S_matrix_[:, :, t : t + t_x]
            mult = numpy.multiply(used_submatrix, polyhedron.tensor_)
            sum = numpy.sum(mult)

            if sum == 0:
                return t

        return self.S_matrix_.shape[t_axis]

    def add_polyhedron_to_schedule(self, polyhedron: polyhedron, t_start: int) -> None:
        """
        Register a polyhedron to the schedule.

        Arguments:
            polyhedron : polyhedron - The polyhedron to register
            t_start : int - The time to register the polyhedron.

        """
        self.S_matrix_ = register_polyhedron(self, t_start, polyhedron)

    def get_task_at(self, unit: int, time: int) -> int:
        """
        Get the scheduled task at a given time on a given unit.

        Arguments:
            unit : int - index of the given unit
            time : int - index of the given time

        Returns:
            int - index of the scheduled task
        """
        return self.S_matrix_[0, unit, time]

    def get_latest_index(self) -> list[int]:
        """
        Get the index of the latest scheduled task.
        """
        latest: list[int] = []
        for u in range(0, self.S_matrix_.shape[u_axis]):
            reversed_matrix = self.S_matrix_[0, u, ::-1]
            latest.append(len(reversed_matrix) - numpy.argmax(reversed_matrix != 0) - 1)

        return latest
    
    def get_latest_index_on_unit(self, u : int) -> list[int]:
        """
        Get the index of the latest scheduled task.

        Arguments:
            u : int - The unit
        """
        reversed_matrix = self.S_matrix_[0, u, ::-1]
        latest = len(reversed_matrix) - numpy.argmax(reversed_matrix != 0) - 1

        return latest
    
    def get_latest_unit(self) -> int:
        """
        Get the unit of the latest scheduled task.
        """
        latest = math.inf
        chosen_unit = -1
        for u in range(0, self.S_matrix_.shape[u_axis]):
            reversed_matrix = self.S_matrix_[0, u, ::-1]
            index = numpy.argmax(reversed_matrix != 0) - 1
            if index < latest and sum(reversed_matrix) != 0:
                chosen_unit = u
                latest = index

        return chosen_unit


    def clear_task(self, unit_id: int) -> None:
        """
        Remove a task completly from the schedule.

        Arguments:
            unit_id : int - Index of the task to be removed.
        """
        self.S_matrix_[self.S_matrix_ == unit_id] = 0

    def get_swapping_tasks(
        self,
        swap_polyhedron: polyhedron,
        all_tasks: list[platform_exports.task_tools.task],
    ) -> tuple[int, list[int]]:
        """
        Return the best cost and the list of task indexes to swapp a polyhedron with.

        Arguments:
            polyhedron : polyhedron - The polyhedron to search a swap for
            all_task : list[task] - A list of all tasks as a dictionary

        Returns:
            tuple[int, list[int]] - Tuple of best cost and associated indexes
        """
        best_cost = math.inf
        best_list: list[int] = []

        # Create a sub matrix of the schedule, cross-product them and then count the hits.
        # Repeat for every position and chose the one with the fewest hits.
        t_x = swap_polyhedron.tensor_.shape[t_axis]
        length_diff = self.S_matrix_.shape[t_axis] - t_x
        for t in range(0, length_diff):
            used_submatrix = self.S_matrix_[:, :, t : t + t_x]
            mult = numpy.multiply(used_submatrix, swap_polyhedron.generate_mask())
            unique_tasks_idx = numpy.unique(mult)
            unique_tasks_idx = unique_tasks_idx[unique_tasks_idx != 0]
            for index in unique_tasks_idx:
                clipped = mult.clip(0, 1)
                cost = numpy.sum(clipped)

                if cost < best_cost:
                    best_cost = cost
                    best_list = unique_tasks_idx

        # If no best cost was found, something went wrong
        if best_cost == math.inf:
            print("No swapping found!")

        return best_cost, [d.get_id() for d in all_tasks if d.get_id() in best_list]


def register_polyhedron(
    schedule: abstract_schedule, t_start: int, polyhedron: polyhedron
) -> numpy.ndarray:
    """
    Register a polyhedron to the schedule. Only works, if the position at the polyhedron is free.

    Arguments:
        schedule : abstract_schedule - The schedule to register the polyhedron to
        t_start : int - The start point of the polyhedron
        polyhedron : polyhedron - The polyhedron to register

    Returns:
        numpy.ndarray - The resulting schedule
    """

    # Get submatrix to add polyhedron to
    t_x = polyhedron.tensor_.shape[t_axis]
    test_submatrix = numpy.take(a=schedule.S_matrix_,indices=range(t_start,t_start+t_x),axis=t_axis,mode="wrap")

    # Check for collisions and raise error, if it is not free
    mult = numpy.multiply(test_submatrix, polyhedron.tensor_)
    sum = numpy.sum(mult)
    if sum != 0:
        raise RuntimeError("Time slot not free! Start: " + str(t_start) + ", unit: " + str(polyhedron.exec_unit_))

    # Insert the new polyhedron.
    inserted_array = numpy.add(polyhedron.tensor_, test_submatrix)

    # Remove old part from array by rolling it to the front...
    rolled_array = numpy.roll(schedule.S_matrix_,-t_start,t_axis)

    # ... deleting the first t_x elements ...
    temporary_array = numpy.delete(rolled_array, range(0, t_x), t_axis)

    # ... insert new polyhedron ...
    rolled_finished_array = numpy.insert(temporary_array, obj=[0], values=inserted_array, axis=t_axis)

    # ... and rolling it back
    finished_array = numpy.roll(rolled_finished_array,t_start,t_axis)

    return finished_array


def print_abstract_schedule_table(table: abstract_schedule, output: io.TextIOWrapper = sys.stdout) -> None:
    """
    Print the schedule table to a file
    """
    # Execution
    for resource in range(table.get_shared_resources()):
        for u in range(0, table.get_no_units()):
            output.write("Resource: " + str(resource) + " Unit: " + str(u) + " |\t")
            for t in range(table.S_matrix_.shape[t_axis]):
                output.write("{:1X}".format(table.S_matrix_[resource][u][t]))
            output.write("\n")
        output.write("------------------------------------\n")


def get_last_first_empty(S_matrix: numpy.array) -> list[int]:
    """
    Get the first empty timeslot for each unit, afterwich no other task is scheduled.

    Arguments:
        S_matrix : numpy.array - The schedule matrix to check.

    Returns:
        list[int] - First timeslot that is empty afterwards per unit.
    """
    last_used_list: list[int] = []
    for u in range(0, S_matrix.shape[u_axis]):
        beta = S_matrix[0][u][:]
        index = numpy.where(beta != 0)
        if numpy.size(index) == 0:
            last_used_list.append(0)
        else:
            last_used_list.append(numpy.max(index) + 1)

    return last_used_list


class abstract_schedule_creator:
    """
    Abstract interface for schedule creation.
    """

    def __init__(self) -> None:
        pass

    def create_schedule(
        self,
        sched_tasks: list[platform_exports.task_tools.task],
        no_units: int,
    ) -> abstract_schedule:
        raise NotImplemented("Abstract Schedule Creator not implemented!")
