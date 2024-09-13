import platform_exports.platform_tools
import platform_exports.task_tools
import platform_exports.unit_tools
import polyhedron_heuristics.block_step
import polyhedron_heuristics.homogeneous_annealing
import polyhedron_heuristics.heterogeneous_annealing
import platform_exports
import polyhedron_heuristics.schedule_tools

SCHED_LENGTH = 250

def main():
    task_builder = platform_exports.task_tools.file_task_builder("examples/task_set_small_with_relations.csv")
    tasks = task_builder.get_tasks()

    unit_builder = platform_exports.unit_tools.file_unit_builder("examples/units.csv")
    units  = unit_builder.get_units()

    #cost_function = polyhedron_heuristics.heterogeneous_annealing.heterogeneous_cost_function(tasks,units,SCHED_LENGTH,max_exec_number=2)
    #creator = polyhedron_heuristics.heterogeneous_annealing.annealing_schedule_creator(cost_function)
    creator = polyhedron_heuristics.block_step.block_step(SCHED_LENGTH)

    schedule = creator.create_schedule(tasks,units)

    with open("schedule3.txt", "w") as f:
        polyhedron_heuristics.schedule_tools.print_abstract_schedule_table(schedule, f)

if __name__ == "__main__":
    main()