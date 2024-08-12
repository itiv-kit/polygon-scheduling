from collections import defaultdict
import io
from math import inf
import math
import pathlib
import csv
import numpy


class task:
    """
    A class to represent an abstract software task.

    Members:
        name_ : str - The name of the task. Used to identify and describe it.
        id_ : int - ID of task. used for indexing and in printing and visual representations.
        exec_time_ : int - The execution time of the task, usually WCET
        path_ : str - A path towards the executable, useful for generating real world schedules
        periodic_deadline_ : int - The deadline of the task. If deadline == -1, the task is non-critical
        access_windows_ : dict[int, list[tuple[int, int]]] - Windows of access to shared resources. Used to prevent contention
        exclusive_windows_ : list[tuple[int, int]] - Windows of exclusive access to all resources. Used to prevent side channel attacks.
        resulution_ : int - The timing resolution of the task. Usefull for minimizing computational effort, when building schedules.
    """

    def __init__(
        self,
        name_str: str,
        id: int,
        exec_time: int,
        file_path: str,
        deadline: int = -1,
        access_windows: dict[int, list[tuple[int, int]]] | None = None,
        exclusive_windows: list[tuple[int, int]] | None = None,
        resulution: int = 1,
    ) -> None:
        self.name_ = name_str
        self.id_ = id
        self.exec_time_ = exec_time
        self.path_ = pathlib.Path(file_path)
        # If no deadline exists, the task is no critical task
        self.criticality_ = deadline > 0
        self.periodic_deadline_ = deadline
        self.access_windows_ = access_windows
        self.exclusive_windows_ = exclusive_windows
        self.resolution_ = resulution

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

    def get_exec_time(self) -> int:
        """
        Return the execution time (WCET) of the task.
        """
        return self.exec_time_

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

    def has_exclusive_window(self) -> bool:
        """
        Return, if the task itself has windows of exclusive execution.
        """
        return not self.exclusive_windows_ is None

    def get_exclusive_windows(self) -> list[tuple[int, int]] | None:
        """
        Return the list of exclusive windows.
        """
        return self.exclusive_windows_

    def has_access_window(self) -> bool:
        """
        Determine, if the task has access windows.
        """
        return not self.access_windows_ is None

    def get_access_windows(self) -> dict[int, list[tuple[int, int]]] | None:
        """
        Return the dictionary specifing the access time to shared resources.
        """
        return self.access_windows_

    def get_resolution(self) -> int:
        """
        Return the timing resolution of the task.
        """
        return self.resolution_

    def fit_to_resolution(self, new_resolution: int) -> None:
        """
        Resize every value to fit the predetermined resolution.

        Arguments:
            new_resolution : int - The used new resolution.
        """

        # Calculate the correction factor between old and new resolution
        correction_factor = self.resolution_ / new_resolution

        # Set execution time and periodinc deadline to closest int
        self.exec_time_ = round(self.exec_time_ * correction_factor)
        self.periodic_deadline_ = round(self.periodic_deadline_ * correction_factor)

        # Resize exclusive execution windows (if applicable)
        if self.exclusive_windows_ != None:
            new_windows: list[tuple[int, int]] = []
            for exclusive_window in self.exclusive_windows_:
                new_windows.append(
                    [
                        int(exclusive_window[0] * correction_factor),
                        int(exclusive_window[1] * correction_factor),
                    ]
                )
            self.exclusive_windows_ = new_windows

        # Resize access execution windows (if applicable)
        if self.access_windows_ != None:
            for acc_window_key in self.access_windows_.keys():
                new_windows: list[tuple[int, int]] = []
                for acc_window in self.access_windows_[acc_window_key]:
                    new_windows.append(
                        [
                            int(acc_window[0] * correction_factor),
                            int(acc_window[1] * correction_factor),
                        ]
                    )
                self.access_windows_[acc_window_key] = new_windows

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

    name_str = "Name"
    exec_time_str = "ExTime"
    path_str = "Path"
    crit_str = "Crit"
    deadline_str = "Deadline"

    excl_windows_str = "ExclWindow"
    excl_window_del_str = ","
    excl_split_str = "/"

    acc_windows_str = "AccWindow"
    acc_window_del_str = ","
    acc_split_str = "/"
    acc_window_unit_str = "AccUnit"

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

    def get_excl_windows(self, line_str: str) -> list[tuple[int, int]] | None:
        """
        Get the windows of exclusive execution.

        Arguments:
            line_str : str - The start and end pairs as a string

        Returns:
            list[tuple[int, int]] - List of exclusive execution start and end points
        """

        # Skip, if line is empty (No windows specified)
        if line_str == None:
            return None

        # Call function with fitting arguments
        return self._get_windows(
            line_str,
            file_task_builder.excl_window_del_str,
            file_task_builder.excl_split_str,
        )

    def get_acc_windows(self, line_str: str) -> list[tuple[int, int]] | None:
        """
        Get the windows of access to share resources. Note: Assignment to resource needs to be done seperately.

        Arguments:
            line_str : str - The start and end pairs as a string

        Returns:
            list[tuple[int, int]] - List of access windows start and end points
        """

        # Skip, if line is empty (No windows specified)
        if line_str == None:
            return None

        # Call function with fitting arguments
        return self._get_windows(
            line_str,
            file_task_builder.acc_window_del_str,
            file_task_builder.acc_split_str,
        )

    def _get_units(self, line_str: str, split_str: str) -> list[int]:
        """
        Get units out of predefined string.

        Arguments:
            line_str: str - String containing a sequence of ints determining the units
            split_str: str - String determining the split between units

        Returns:
            list[int] - A sorted list of ints determining a unit
        """
        splitted = line_str.strip().split(split_str)
        units = [int(character) for character in splitted if character != ""]
        return units

    def get_acc_units(self, line_str: str) -> list[int] | None:
        """
        Return the list of units for the access windows.

        Arguments:
            line_str : str - String to extract units from
        """
        if line_str == None:
            return None

        # Call function with corresponding arguments
        return self._get_units(line_str, file_task_builder.acc_window_del_str)

    def from_file(self, file: io.TextIOWrapper) -> list[task]:
        """
        Functional center of class. Gets TextIOWrapper as input and generates a list of tasks from that.

        Arguments:
            file : io.TextIOWrapper - Input stream into the function

        Returns:
            list[task] - A list of tasks represented by the input string
        """

        # Open file as CSV
        csv_reader = csv.DictReader(file, delimiter=file_task_builder.del_, quotechar=file_task_builder.quote_)
        task_list = []
        task_id = 1
        for line in csv_reader:

            # Get window lists and dict
            excl_windows = self.get_excl_windows(line[file_task_builder.excl_windows_str])
            acc_windows = self.get_acc_windows(line[file_task_builder.acc_windows_str])
            acc_window_units = self.get_acc_units(line[file_task_builder.acc_window_unit_str])

            acc_dict = None
            if acc_windows is not None and acc_window_units is not None:
                if len(acc_windows) != len(acc_window_units):
                    raise RuntimeError("Length of units and windows missmatch!")
                
                acc_dict : dict[int, list[tuple[int, int]]] = defaultdict(list)
                for index in range(len(acc_windows)):
                    acc_dict[acc_window_units[index]].append(acc_windows[index])

            # Call task constructor
            current_task = task(
                line[file_task_builder.name_str],
                task_id,
                int(line[file_task_builder.exec_time_str]),
                line[file_task_builder.path_str],
                int(line[file_task_builder.deadline_str]),
                acc_dict,
                excl_windows,
            )

            # Append task to list and increment ID
            task_list.append(current_task)
            task_id = task_id + 1
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
    deadline: int = inf
    for task in tasks:
        if task.is_critical() and task.get_periodic_deadline() < deadline:
            deadline = task.get_periodic_deadline()
    return deadline


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

    for task in tasks:

        # Round every execution time up to the nearest platform minimum
        round_up = math.ceil(task.get_exec_time() / platform_minimum) * platform_minimum
        exec_times.append(round_up)

        if task.get_exclusive_windows() is not None:

            # Iterate over all exclusive windows and add the numbers to the list.
            for exclusive_windows in task.get_exclusive_windows():
                exec_times.extend(exclusive_windows)

        if task.get_access_windows() is not None:
            for acc_key in task.get_access_windows().keys():
                for acc_tuple in task.get_access_windows()[acc_key]:
                    exec_times.extend(acc_tuple)

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
        length: int - The length of the polyhedron
        task_id: int - The ID of the corresponding task
        acc_windows: dict[int, list[tuple[int, int]]] - List of access windows sorted by shared resource
        sec_windows_: list[tuple[int, int]] - List security windows

        tensor_ : numpy.array - Shape representation of task
    """

    def __init__(
        self,
        no_units: int,
        exec_unit: int,
        length: int,
        task_id: int,
        acc_windows: dict[int, list[tuple[int, int]]] | None = None,
        sec_windows: list[tuple[int, int]] | None = None,
        no_shared_resources: int = 0,
    ) -> None:
        # Book keeping
        self.no_units_ = no_units
        self.exec_unit_ = exec_unit
        self.length_ = length
        self.task_id_ = task_id
        self.acc_windows_ = acc_windows
        self.sec_windows_ = sec_windows

        self.tensor_ = numpy.zeros(shape=[no_shared_resources + 1, self.no_units_, self.length_])

        # Main execution
        for t in range(0, self.length_):
            self.tensor_[0][self.exec_unit_][t] = self.task_id_

        # Access window
        if self.acc_windows_ is not None:
            for resource_index in self.acc_windows_.keys():
                windows = self.acc_windows_[resource_index]
                for u in range(0, self.no_units_):
                    for acc_window in windows:
                        for t in range(acc_window[0], acc_window[1]):
                            self.tensor_[resource_index][u][t] = self.task_id_

        # Security window
        if self.sec_windows_ is not None:
            for u in range(0, self.no_units_):
                for sec_windows in self.sec_windows_:
                    for t in range(sec_windows[0], sec_windows[1]):
                        self.tensor_[0][u][t] = self.task_id_
                        self.tensor_[1][u][t] = self.task_id_

    def get_task_id(self) -> int:
        """
        Returns the task ID.
        """
        return self.task_id_

    def get_length(self) -> int:
        """
        Returns the polyhedron length.
        """
        return self.length_

    def get_tensor(self) -> numpy.array:
        """
        Returns raw view on the tensor.
        """
        return self.tensor_

    def __str__(self) -> str:
        """
        Returns a string representation of the tensor, ready for use in print() statements or similar
        """
        string = "Polyhedron: T: "
        string.append(self.task_id_ + " ")
        string.append("U: ")
        string.append(str(self.exec_unit_) + "/" + str(self.no_units_) + "\n")
        string.append("Shape Execution:\n")

        for t in range(0, self.length_):
            for u in range(0, self.no_units_):
                string.append(str(self.tensor_[0][u][t]) + " ")
            string.append("\n")

        string.append("Shape Access:\n")

        for acc_unit, _ in self.acc_windows_:
            for t in range(0, self.length_):
                for u in range(0, self.no_units_):
                    string.append(str(self.tensor_[acc_unit][u][t]) + " ")
            string.append("\n")

        return string


def create_task_polyhedrons(input_task: task, no_units: int, shared_resources: int = 0) -> list[polyhedron]:
    """
    Create all possible polyhedrons from a task for HOMOGENEOUS execution units and the total amount of units.
    """
    polyhedron_list: list[polyhedron] = []

    for u in range(0, no_units):
        new_polyhedron = polyhedron(
            no_units,
            u,
            input_task.get_exec_time(),
            input_task.get_id(),
            input_task.get_access_windows(),
            input_task.get_exclusive_windows(),
            shared_resources,
        )
        polyhedron_list.append(new_polyhedron)

    return polyhedron_list
