import platform_exports.platform_tools
import platform_exports.task_tools
import platform_exports.unit_tools
import polyhedron_heuristics.block_step
import polyhedron_heuristics.heterogeneous_annealing
import platform_exports
import polyhedron_heuristics.schedule_tools

SCHED_LENGTH = 2000

def main():
    task_builder = platform_exports.task_tools.file_task_builder("examples/task_set_small.csv")
    tasks = task_builder.get_tasks()

    unit_builder = platform_exports.unit_tools.file_unit_builder("examples/units.csv")
    units  = unit_builder.get_units()

    cost_function = polyhedron_heuristics.heterogeneous_annealing.soft_cost_function(tasks,len(units),SCHED_LENGTH,max_exec_number=2)
    creator = polyhedron_heuristics.heterogeneous_annealing.annealing_schedule_creator(cost_function)

    schedule = creator.create_schedule(tasks,len(units))

    with open("schedule3.txt", "w") as f:
        polyhedron_heuristics.schedule_tools.print_abstract_schedule_table(schedule, f)

if __name__ == "__main__":
    main()