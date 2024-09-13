from collections import defaultdict
import io
from math import ceil, inf
import math
import pathlib
import csv
import numpy
from enum import Enum
from platform_exports.unit_tools import unit

t_axis = 2
u_axis = 1
acc_axis = 0


class task_relation:
    """
    Class to specify the relation between two tasks. The window is given as the maximum and minimum difference between the two tasks.
    """

    def __init__(self, window_start: int, window_end: int) -> None:
        self.window_start_ = window_start
        self.window_end_ = window_end

    def get_window_start(self) -> int:
        """
        Return the start of the window.
        """
        return self.window_start_

    def get_window_end(self) -> int:
        """
        Return the end of the window.
        """
        return self.window_end_

    def reset_values(self, window_start: int, window_end: int) -> None:
        """
        Reset the values (useful for changing resolution)
        """
        self.window_start_ = window_start
        self.window_end_ = window_end


class execution_window:
    """
    Class representing parallel execution on other resources including start time, stop time, resource index and number of blocked resources of type "resource_index".
    If the corresponding task needs parallel execution (i.e. blocks another execution resource) this is signified by resource_index = 0 and a number of blocked_resources != -1.
    If all resources of type resource_idx are blocked in the timing period, this is marked with blocked_resources = -1. If i.e. 2 are blocked, this is specified with
    blocked_resources = 2.
    """

    def __init__(self, start_time: int, stop_time: int, resource_idx: int, blocked_resources: int = -1) -> None:
        self.start_ = start_time
        self.stop_ = stop_time
        if resource_idx == 0:
            raise RuntimeError("Shared resource cannot be 0!")

        self.resource_idx_ = resource_idx
        self.blocked_resources_ = blocked_resources

    def get_start(self) -> int:
        """
        Returns the start of the execution window.
        """
        return self.start_

    def get_stop(self) -> int:
        """
        Returns the end of the execution window.
        """
        return self.stop_

    def get_borders(self) -> tuple[int, int]:
        """
        Returns both the start and the end of the execution window.
        """
        return self.start_, self.stop_

    def set_borders(self, start_time: int, stop_time: int) -> None:
        self.start_ = start_time
        self.stop_ = stop_time

    def get_resource(self) -> int:
        """
        Get the resource of the execution window.
        """
        return self.resource_idx_

    def fills_all_resources(self) -> bool:
        """
        Return, if the access window fills all resources of type resource_idx_ or is limited to less.
        """
        return self.blocked_resources_ == -1

    def get_number_of_blocked_resources(self) -> int:
        """
        Return the number of blocked resources of type resource_idx.
        """
        return self.blocked_resources_


class task:
    """
    A class to represent an abstract software task.

    Members:
        name_ : str - The name of the task. Used to identify and describe it.
        id_ : int - ID of task. used for indexing and in printing and visual representations.
        exec_time_ : dict[int,int] - The execution time of the task on the given unit, usually WCET
        path_ : str - A path towards the executable, useful for generating real world schedules
        periodic_deadline_ : int - The deadline of the task. If deadline == -1, the task is non-critical
        access_windows_ : dict[int, execution_window] - Windows of access necessities to other resources depending on the execution on unit type "key".
        resulution_ : int - The timing resolution of the task. Usefull for minimizing computational effort, when building schedules.
    """

    def __init__(
        self,
        name_str: str,
        id: int,
        exec_time: dict[int,int],
        file_path: str,
        deadline: int = -1,
        access_windows: dict[int,list[execution_window]] | None = None,
        task_relations: dict["task", task_relation] | None = None,
        resulution: int = 1,
    ) -> None:
        self.name_ = name_str
        self.id_ = id
        # Execution time on non supported units is defined as infinite
        self.exec_times_ = exec_time
        self.path_ = pathlib.Path(file_path)
        # If no deadline exists, the task is no critical task
        self.criticality_ = deadline > 0
        self.periodic_deadline_ = deadline
        self.access_windows_ = access_windows
        self.task_relations_ = task_relations
        self.resolution_ = resulution

    def register_task_relations(self, relations: dict['task', task_relation]) -> None:
        """
        Register the task relations to this task.
        """
        self.task_relations_ = relations

    def get_id(self) -> int:
        """
        Return the task ID.
        """
        return self.id_

    def get_name(self) -> str:
        """
        Return the task name.
        """
        return self.name_
    
    def get_exec_time(self, unit_type : int) -> int:
        """
        Return the execution time of the task on unit type unit_idx.
        """
        return int(self.exec_times_[unit_type])

    def get_path(self) -> str:
        """
        Return the path of the executable.
        """
        return self.path_

    def is_critical(self) -> bool:
        """
        Return flag, if task has a deadline.
        """
        return self.criticality_

    def get_periodic_deadline(self) -> int:
        """
        Get the periodic deadline for the task.
        """
        return self.periodic_deadline_

    def has_access_window(self) -> bool:
        """
        Determine, if the task has access windows.
        """
        return not self.access_windows_ is None

    def get_access_windows(self) -> dict[int,list[execution_window]] | None:
        """
        Return the dictionary specifing the access time to shared resources.
        """
        return self.access_windows_

    def get_access_window(self, resource: int) -> list[execution_window]:
        """
        Get the access windows to a shared resource with a defined index.
        """
        return [exec_window for exec_window in self.access_windows_ if exec_window.get_resource() == resource]

    def get_resolution(self) -> int:
        """
        Return the timing resolution of the task.
        """
        return self.resolution_

    def get_relations(self) -> dict['task', task_relation]:
        """
        Returns all task relations.
        """
        return self.task_relations_

    def get_relation(self, other_task: 'task') -> task_relation:
        """
        Returns the relation to another task.
        """
        return self.task_relations_[other_task]
    
    def get_supported_exec_unit_indeces(self) -> list[int]:
        """
        Returns the index of all supported execution units.
        """
        return list(self.exec_times_.keys())

    def fit_to_resolution(self, new_resolution: int) -> None:
        """
        Resize every value to fit the predetermined resolution.

        Arguments:
            new_resolution : int - The used new resolution.
        """

        # Calculate the correction factor between old and new resolution
        correction_factor = self.resolution_ / new_resolution

        # Set execution time and periodinc deadline to closest int
        new_exec_times : dict[int,int] = {}
        for exec_time_key in self.exec_times_.keys():
            new_exec_times[exec_time_key] = ceil(self.exec_times_[exec_time_key] * correction_factor)
        self.exec_times_ = new_exec_times
        self.periodic_deadline_ = ceil(self.periodic_deadline_ * correction_factor)

        # Resize access execution windows (if applicable)
        if self.access_windows_ != None:
            for acc_window_key in self.access_windows_.keys():
                for acc_window in self.access_windows_[acc_window_key]:
                    acc_start = acc_window.get_start()
                    acc_stop = acc_window.get_stop()
                    acc_window.set_borders(ceil(acc_start * correction_factor), ceil(acc_stop * correction_factor))

        if self.task_relations_ is not None:
            for rel_key in self.task_relations_.keys():
                relation = self.task_relations_[rel_key]
                relation.reset_values(
                    ceil(relation.get_window_start() * new_resolution), ceil(relation.get_window_end() * new_resolution)
                )

        # Set the new resolution
        self.resolution_ = new_resolution


class task_builder:
    """
    Abstract class to generate tasks. Children classes include task builder for test and from file.
    """

    def get_tasks(self) -> list[task]:
        raise NotImplementedError


class file_task_builder(task_builder):
    """
    Class to read description from a CSV file and build the corresponding tasks.

    Members:
        file_str : str - Path towards file.
    """

    # Definitions
    del_ = ";"
    quote_ = "\n"

    list_del_str = ","
    list_split_str = "/"

    id_str = "ID"
    res_str = "Res"

    name_str = "Name"
    exec_time_str = "ExTime"
    exec_unit_str = "ExUnit"
    path_str = "Path"
    crit_str = "Crit"
    deadline_str = "Deadline"
    res_string = "Resolution"

    relation_window_str = "RelWindows"
    relation_tasks_str = "RelTasks"

    acc_windows_str = "AccWindow"
    acc_window_unit_str = "AccUnit"
    acc_window_number_str = "AccNumber"
    acc_val_exec_unit_str = "AccValidExec"

    def __init__(self, file_str: str) -> None:
        self.file_str = file_str

    def from_file_str(self, file_str: str) -> list[task]:
        """
        Open a file and return the tasks inside.

        Arguments:
            file_str : str - The path of towards the file

        Returns:
            list[task] - tasks specified inside the file
        """
        with open(file_str, "r") as f:
            task_list = self.from_file(f)
        return task_list

    def _get_windows(self, line_str: str, delimiter_str: str, split_str: str) -> list[tuple[int, int]]:
        """
        Get exclusive or access windows depending on a given string.

        Arguments:
            line_string : str - The string to read from
            delimiter_str : str - The string where the windows are split
            split_str : str - The string to split start and end of window

        Return:
            list[tuple[int, int]] - List of tuples of start and end of each window
        """

        # Split complete string into a list of strings, each determining one window
        splitted = line_str.strip().split(delimiter_str)
        windows: list[tuple[int, int]] = []

        # Iterate over every splitted string
        for pair in splitted:
            if pair == "":
                return None

            # Split the string into two halfs
            pair_split = pair.split(split_str)

            # Append determined start and end points
            windows.append([int(pair_split[0]), int(pair_split[1])])
        return windows

    def get_task_from_id(self, id : int, tasks : list[task]) -> task:
        """
        Return the task, if the id is given.
        """
        return [id_task for id_task in tasks if id_task.get_id() == id][0]


    def _get_list(self, line_str: str, split_str: str = list_split_str) -> list[int]:
        """
        Get a list of items out of predefined string.

        Arguments:
            line_str: str - String containing a sequence of ints determining the items
            split_str: str - String determining the split between items

        Returns:
            list[int] - A sorted list of ints determining a unit
        """
        splitted = line_str.strip().split(split_str)
        items = [int(character) for character in splitted if character != ""]
        return items

    def get_valid_units_for_windows(self, line_str : str) -> list[list[int]]:
        """
        Return a list of lists signifying the valid units for the given access windows
        """
        lists : list[list[int]] = []
        splitted = line_str.strip().split(self.list_del_str)
        for splitted_elem in splitted:
            lists.append(self._get_list(splitted_elem,self.list_split_str))

        return lists

    def get_exec_times(self, exec_line_str: str, exec_units_str: int) -> dict[int, int]:
        """
        Return the dict of execution times depending of possible execution units.
        """
        units = self._get_list(exec_units_str, file_task_builder.list_del_str)
        times = self._get_list(exec_line_str, file_task_builder.list_del_str)

        if len(units) != len(times):
            raise RuntimeError("Length of units and exec times missmatch!")

        resulting_dict = dict(zip(units, times))

        return resulting_dict

    def from_file(self, file: io.TextIOWrapper) -> list[task]:
        """
        Functional center of class creation. Gets TextIOWrapper as input of csv file and generates a list of tasks from that.

        Arguments:
            file : io.TextIOWrapper - Input stream into the function

        Returns:
            list[task] - A list of tasks represented by the input string
        """

        # Open file as CSV
        csv_reader = csv.DictReader(file, delimiter=file_task_builder.del_, quotechar=file_task_builder.quote_)
        task_list: list[task] = []

        # Create list to save the task indexes inside
        relation_tuples: dict[task, list[tuple[int,int]]] = {}
        relation_tasks_ids : dict[task,list[int]] = {}

        for line in csv_reader:
            # Get the exec times
            exec_times = self.get_exec_times(line[file_task_builder.exec_time_str],line[file_task_builder.exec_unit_str])

            # Get window lists and dicts
            acc_windows = self._get_windows(line[file_task_builder.acc_windows_str],file_task_builder.list_del_str,file_task_builder.list_split_str)
            acc_window_units = self._get_list(line[file_task_builder.acc_window_unit_str], file_task_builder.list_del_str)
            acc_numbers = self._get_list(line[file_task_builder.acc_window_number_str], file_task_builder.list_del_str)
            acc_valid_units = self.get_valid_units_for_windows(line[file_task_builder.acc_val_exec_unit_str])

            acc_dict = None
            # Check if the acc windows have time, shared resource and execution units for what they are valid available.
            if acc_windows is not None and acc_window_units is not None and acc_numbers is not None:
                if (
                    len(acc_windows) != len(acc_window_units)
                    or len(acc_numbers) != len(acc_windows)
                    or len(acc_numbers) != len(acc_valid_units)
                ):
                    raise RuntimeError("Length of units, windows and numbers missmatch!")

                acc_dict: dict[int, list[execution_window]] = defaultdict(list)

                for index in range(len(acc_windows)):
                    new_acc_window = execution_window(acc_windows[index][0], acc_windows[index][1], acc_window_units[index], acc_numbers[index])
                    for valid_unit in acc_valid_units[index]:
                        acc_dict[valid_unit].append(new_acc_window)

            # Read the rest
            task_name = line[file_task_builder.name_str]
            task_id = int(line[file_task_builder.id_str])
            task_path = line[file_task_builder.path_str] 
            task_deadline = int(line[file_task_builder.deadline_str]) if line[file_task_builder.deadline_str] != "" else -1
            task_resolution = int(line[file_task_builder.res_str]) if line[file_task_builder.res_str] != "" else 1

            # Call task constructor
            current_task = task(
                task_name,
                task_id,
                exec_times,
                task_path,
                task_deadline,
                acc_dict,
                None,
                task_resolution,
            )

            # Append task
            task_list.append(current_task)

            # Save the relations in a dictionary
            relation_tuples[current_task] = self._get_windows(line[file_task_builder.relation_window_str],file_task_builder.list_del_str,file_task_builder.list_split_str)
            relation_tasks_ids[current_task] = self._get_list(line[file_task_builder.relation_tasks_str], file_task_builder.list_split_str)

        # Append task relations
        for current_task in task_list:

            # Skip, if the task has no associated IDs
            if relation_tasks_ids[current_task]: # If the list is not empty
                current_relations = [task_relation(relation_tuple[0], relation_tuple[1]) for relation_tuple in relation_tuples[current_task]]
                relation_tasks = [self.get_task_from_id(id_idx, task_list) for id_idx in relation_tasks_ids[current_task]]
                relation_dict = dict(zip(relation_tasks,current_relations))
                current_task.register_task_relations(relation_dict)

        # Return all tasks
        return task_list

    def get_tasks(self) -> list[task]:
        """
        Call to the generator function.
        """
        return self.from_file_str(self.file_str)


def get_minimum_deadline(tasks: list[task]) -> int:
    """
    Returns the minimum deadline from all tasks.

    Arguments:
        tasks : list[task] - The list of tasks

    Returns:
        int - The shortest deadline
    """
    return min([t.get_periodic_deadline() for t in tasks if t.is_critical()])


def get_timing_resolution(tasks: list[task], platform_minimum: int = 1) -> int:
    """
    Calculate the greatest common denominator of all timing specifications (Accesss windows, Security windows, Execution time).
    Also includes the possible timing resolution of the platform.

    Arguments:
        tasks : list[task] - A list of all tasks to create the gcd from
        platform_minimum : int - the minimum resolution of the execution platform

    Returns:
        int - The gcd of all timing specifications and the platform
    """
    exec_times: list[int] = []
    exec_times.append(platform_minimum)

    for task in tasks:
        # Round every execution time up to the nearest platform minimum

        for possible_execution_unit in task.get_supported_exec_unit_indeces():
            exec_times.append(task.get_exec_time(possible_execution_unit))

        if task.get_access_windows() is not None:
            for acc_key in task.get_access_windows().keys():
                for acc_tuple in task.get_access_windows()[acc_key]:
                    exec_times.extend(acc_tuple.get_borders())

    # Calculate gcd of all values
    exec_time_gcd = math.gcd(*exec_times)

    return exec_time_gcd


class polyhedron:
    """
    Polyhedron class representing a task in the form of a 3D polyhedron, where the first dimension signifies the shared
    ressource, the second one signifies the execution unit and the third dimension signifies time. The representation
    is done in the form of a numpy.array member, filled with the index or a 0.

    Members:
        no_units_ : int - The number of available execution units
        exec_unit: int - The chosen execution unit
        task_id: int - The ID of the corresponding task

        tensor_ : numpy.array - Shape representation of task
    """
    def __init__(
        self,
        no_units: int,
        exec_unit: int,
        length: int,
        task_id: int,
        acc_windows: list[execution_window] | None = None,
        no_shared_resources: int = 0,
    ) -> None:
        # Book keeping
        self.exec_unit_ = exec_unit
        self.task_id_ = task_id

        self.tensor_ = numpy.zeros(shape=[no_shared_resources + 1, no_units, length])

        # Main execution
        for t in range(0, length):
            self.tensor_[0][self.exec_unit_][t] = self.task_id_

        # Access window
        if acc_windows is not None:
            for acc_window in acc_windows:
                resource_index = acc_window.get_resource()

                if resource_index == 0:
                    # resource_index != 0 to keep execution plane empty!
                    raise RuntimeError("Resource index can't be 0!")  

                for t in range(acc_window.get_start(), acc_window.get_stop()):
                    self.tensor_[resource_index,:,t] = self.task_id_  # <- resource_index != 0 to keep execution plane empty!

    def get_task_id(self) -> int:
        """
        Returns the task ID.
        """
        return self.task_id_

    def get_no_units(self) -> int:
        """
        Return the number of units.
        """
        return self.tensor_.shape[u_axis]

    def get_length(self) -> int:
        """
        Returns the polyhedron length.
        """
        return self.tensor_.shape[t_axis]

    def get_tensor(self) -> numpy.array:
        """
        Returns raw view on the tensor.
        """
        return self.tensor_

    def get_resource_number(self) -> int:
        """
        Returns the amount of shared resources
        """
        return self.tensor_.shape[acc_axis]

    def generate_mask(self) -> numpy.array:
        """
        Return a mask of the current polyhedron where 0 stays 0 and every value is truncated to 1
        """
        this_tensor = numpy.copy(self.get_tensor())
        this_tensor[this_tensor > 0] = 1

        return this_tensor


    def __str__(self) -> str:
        """
        Returns a string representation of the tensor, ready for use in print() statements or similar
        """
        string = "Polyhedron: T: "
        string.append(self.task_id_ + " ")
        string.append("U: ")
        string.append(str(self.exec_unit_) + "/" + str(self.get_no_units()) + "\n")
        string.append("Shape Execution:\n")

        for t in range(0, self.get_length()):
            for u in range(0, self.get_no_units()):
                string.append(str(self.tensor_[0][u][t]) + " ")
            string.append("\n")

        string.append("Shape Access:\n")

        for acc_unit, _ in self.acc_windows_:
            for t in range(0, self.get_length()):
                for u in range(0, self.get_no_units()):
                    string.append(str(self.tensor_[acc_unit][u][t]) + " ")
            string.append("\n")

        return string
    
    
def create_task_polyhedrons(input_task: task, shared_resources: int, unit_list : list[unit]) -> list[polyhedron]:
    """
    Create all possible polyhedrons from a task for HOMOGENEOUS execution units and the total amount of units.
    """
    polyhedron_list: list[polyhedron] = []

    for u in unit_list:
        if u.get_type() in input_task.get_supported_exec_unit_indeces():
            used_access_windows = None if input_task.get_access_windows() is None else input_task.get_access_windows()[u.get_type()]
            new_polyhedron = polyhedron(
                len(unit_list),
                unit_list.index(u),
                input_task.get_exec_time(u.get_type()),
                input_task.get_id(),
                used_access_windows,
                shared_resources
            )
        polyhedron_list.append(new_polyhedron)
        
    return polyhedron_list
