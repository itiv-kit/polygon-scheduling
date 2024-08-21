from collections import defaultdict
from math import ceil
import numpy
import random
import platform_exports.task_tools


# Taken with inspiration from this Stack overflow answer:
# https://stackoverflow.com/a/3590105
def sum_constraint_random(amount, total) -> list[int]:
    dividers = sorted(random.sample(range(1, total), amount - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

def get_task_borders(amount : int, max : int, exclude : list[int] = []) -> list[int]:
    all_borders = random.sample(range(1,max-1), amount -1)
    
    for exclude_elem in exclude:
        all_borders.remove(exclude_elem)

    while len(all_borders) < amount - 1:
        to_add = random.randint(1,max -1)
        if to_add not in exclude:
            all_borders.append(to_add)

    all_borders.extend([0,max])
    all_borders.sort()
    return all_borders


def create_test_tasks(
    schedule_length: int,
    no_tasks: int,
    sec_windows: int = 0,
    acc_windows: int = 0,
    acc_window_resources : int = 1,
    no_units: int = 4,
    timing_slack: int = 10,
) -> list[platform_exports.task_tools.task]:
    """
    Main function for generating task test cases for generating schedule.
    This function generates random task sets given the arguments.

    Arguments:
        schedule_length: int - The length of the schedule
        no_tasks: int - The total number of tasks per unit
        sec_windows: int - The number of security windows
        acc_windows: int - The number of access windows
        acc_window_resources : int - The number of accelerated resources
        units : int - The number of units
        timing_slack: int - The given slack
    """

    # Incorporate slack into schedule length
    slack_schedule_length = int(schedule_length * (100 - timing_slack) / 100)
    
    # Generate time slot starts of access windows and shared resource
    acc_positions = []
    acc_units = []
    if acc_windows != 0:
        acc_positions = random.sample(range(0,slack_schedule_length), acc_windows)
        acc_units = [i % acc_window_resources + 1 for i in range(acc_windows)]

    sec_windows_starts = []
    if sec_windows != 0:
        # Generate security window positions only allow every forth position
        sec_windows_starts = random.sample(range(1,int(slack_schedule_length - 1 / 4)), sec_windows)
        sec_windows_starts.sort()
        sec_windows_starts = [4 * window for window in sec_windows_starts]

        for sec_window_start in acc_positions:
            if sec_window_start in acc_positions:
                acc_positions.remove(sec_window_start)
                acc_positions.append((sec_window_start + 1) % (slack_schedule_length - 1))
        
    # Create the task borders for every unit
    task_borders : dict[int,list[int]] = {}
    for i in range(no_units):
        task_borders[i] = get_task_borders(no_tasks - sec_windows, slack_schedule_length, sec_windows_starts)
        for sec_window_start in sec_windows_starts:
            task_borders[i].append(sec_window_start + 1)
        task_borders[i].sort()
        

    # Create the tasks
    task_id = 0
    tasks: list[platform_exports.task_tools.task] = []
    
    # Determine which unit (and with that task) holds the sec_windows
    sec_window_units : list[int] = []
    for i in range(len(sec_windows_starts)):
        sec_window_units.append(random.randint(0,no_units))

    # Determine which unit (and with that task) holds the acc_windows
    acc_window_units : list[int] = []
    for i in range(len(acc_positions)):
        acc_window_units.append(random.randint(0,no_units))

    for unit in range(no_units):
        for t in range(len(task_borders[unit])-1):
            current_start_time = task_borders[unit][t]
            current_end_time = task_borders[unit][t + 1]
            current_task_length = current_end_time - current_start_time

            # Create sec window, if it is executed right before the current task
            current_sec_windows : list[tuple[int,int]] = []
            for sec_idx in range(len(sec_windows_starts)):
                current_sec_window = sec_windows_starts[sec_idx]
                if current_start_time == current_sec_window + 1 and sec_window_units[sec_idx] == unit:
                    current_sec_windows.append((0,1))

            # Create acc windows, if they are executed on this unit during this time
            current_acc_windows : dict[int,list[tuple[int,int]]] = defaultdict(list)
            for acc_idx in range(len(acc_positions)):
                current_acc_win = acc_positions[acc_idx]
                if current_start_time <= current_acc_win and current_acc_win <= current_end_time and acc_window_units[acc_idx]:
                    current_acc_windows[acc_units[acc_idx]].append((current_acc_win-current_start_time,current_acc_win-current_start_time+1))

            # Create name based on ID, unit, index and start time
            current_name = "task-ID" + str(task_id) + "-u" + str(unit) + "-i" + str(t) + "-st" + str(current_start_time)
            new_task = platform_exports.task_tools.task(current_name,task_id,current_task_length,"/test/",schedule_length,current_acc_windows,current_sec_windows)
            tasks.append(new_task)

            task_id += 1

    return tasks
