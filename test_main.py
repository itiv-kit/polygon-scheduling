import polyhedron_heuristics.block_step
import polyhedron_heuristics.heterogeneous_annealing
import test
import polyhedron_heuristics.schedule_tools
import test.test_heuristic
import time

def test_one(length : int, timing_slack : int, no_tasks:int, no_units : int, no_sec_windows : int, no_acc_windows : int, no_acc_units :int) -> tuple[float, int]:
    tasks = test.test_heuristic.create_test_tasks(
        length,
        no_tasks=no_tasks,
        acc_windows=no_acc_windows,
        acc_window_resources=no_acc_units,
        sec_windows=no_sec_windows,
        no_units=no_units,
        timing_slack=timing_slack,
    )

    index = 1
    for t in tasks:
        t.id_ = index
        index += 1

    schedule_creator = polyhedron_heuristics.block_step.block_step()
    
    start_time = time.time()
    schedule = schedule_creator.create_schedule(tasks, no_units)
    end_time = time.time()

    max_length = polyhedron_heuristics.schedule_tools.get_last_first_empty(schedule.S_matrix_)

    return end_time - start_time, max(max_length), 0

def main():
    test_one(1000,40,1,4,0,2,1)

if __name__ == "__main__":
    main()
