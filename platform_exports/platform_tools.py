import io
import math
import sys
import numpy
import platform_exports.task_tools
import platform_exports.unit_tools
import polyhedron_heuristics

class platform_schedule:
    def __init__(
        self,
        timing_resolution: int,
        tasks: list[platform_exports.task_tools.task],
        units: list[platform_exports.unit_tools.unit],
        S_matrix: numpy.ndarray | None = None,
        periodic_deadline: int | None = None,
    ):
        self.timing_resolution_ = timing_resolution
        self.tasks_ = tasks
        self.units_ = units
        self.periodic_deadline_ = periodic_deadline

        if S_matrix is None:
            self.valid_ = False
            self.S_matrix_ = None
        else:
            self.valid_ = True
            self.S_matrix_ = S_matrix

    def register_table(self, table: numpy.ndarray) -> None:
        self.S_matrix_ = table
        self.valid_ = True

    def get_schedule_matrix(self) -> numpy.ndarray:
        if not self.valid_ or self.S_matrix_ is None:
            raise RuntimeError("Schedule not completed!")

        return self.S_matrix_

    def get_periodic_deadline(self) -> int | None:
        return self.periodic_deadline_

    def get_timing_resolution(self) -> int:
        return self.timing_resolution_

    def get_no_timeslots(self) -> int:
        if not self.valid_ or self.S_matrix_ is None:
            raise RuntimeError("Schedule not completed!")

        return self.S_matrix_.shape[polyhedron_heuristics.schedule_tools.t_axis]

    def get_tasks(self) -> list[platform_exports.task_tools.task]:
        return self.tasks_

    def get_units(self) -> list[platform_exports.unit_tools.unit]:
        return self.units_

    def is_valid(self) -> bool:
        return self.valid_


def print_schedule_table(
    table: platform_schedule, output: io.TextIOWrapper = sys.stdout
) -> None:
    if not table.is_valid():
        output.write("Schedule: ")

    tasks = table.get_tasks()
    units = table.get_units()
    schedule_matrix = table.get_schedule_matrix()

    output.write(
        "Schedule Table S^: Period: "
        + str(table.get_periodic_deadline())
        + ". Timing resolution: "
        + str(table.get_timing_resolution())
        + ".\n"
    )
    output.write("Unit |\t Tasks\n")

    for unit in units:
        output.write(unit.get_name()[0:5] + " |\t")

        for i in range(table.get_no_timeslots()):
            scheduled_task = schedule_matrix[units.index(unit)][i]
            if scheduled_task is polyhedron_heuristics.schedule_tools.empty_symbol:
                scheduled_mark = "X"
            else:
                scheduled_mark = tasks.index(scheduled_task)

            output.write(str(scheduled_mark) + "\t")

        output.write("\n")
