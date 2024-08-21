# TDMA Schedule Creation Library

Accompanying open source code for the paper Automated Polyhedron-based TDMA Schedule Design for Predictable Mixed-Criticality MPSoCs

In this work, we introduce an algorithm creating scheduling tables for TDMA schedulers that support isolation mechanisms in multicore systems. Using a constructive approach, the resulting scheduling tables support mechanisms for parallel access to shared resources and windows for exclusive execution.

Furthermore, a simulated annealing approach is presented for multiple executions per TDMA table in the file ```heterogeneous_annealing.py```.

## Structure
```
- examples
    | - tasks.csv : An example task specification
    | - units.csv : An example unit specification
- platform_exports
    | - platform_tools.py : Platform dependent schedule representation
    | - task_tools.py : Task and Polyhedron classes and task builder
    | - unit_tools.py : Unit class and builder
- polyhedron_heuristics:
    | - block_step.py : Main implementation of Block-Step Algorithm
    | - heterogeneous_annealing.py : Implementation of simulated annealing and corresponding cost function
    | - schedule_tools.py : Abstract schedule representation
- test
    | - test_heuristic.py : Class to generate random task sets for testing
- main.py : Example use for task and unit read in
- test_main.py : Example use of test generation and usage
```
## Citation
If you use this, please cite:

```@inproceedings{bawatna2022,
  title = {Automated Polyhedron-based TDMA Schedule Design for Predictable Mixed-Criticality MPSoCs},
  booktitle = {2024 27th {{Euromicro Conference}} on {{Digital System Design}} ({{DSD}})},
  author = {Stammler, Matthias and Schade, Florian and Becker, Juergen},
  year = {2023},
  month = aug,
  publisher = {IEEE},
  address = {Paris, France}
}
```
