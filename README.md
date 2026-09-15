
<br>
<h1 align="center">Optimal Multi-Agent Path Finding in Continuous Time</h1>
<br>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=SJwGlGYAAAAJ&hl=en">Alvin Combrink</a>, 
    <a href="https://scholar.google.com/citations?user=souzKXIAAAAJ&hl=en">Sabino Franceso Roselli</a>, 
    and <a href="https://scholar.google.com/citations?user=Vv2dq8EAAAAJ&hl=en&oi=ao">Martin Fabian</a>.
  </p>
  <br>
<br>

This is the official repository for the publication **Optimal Multi-Agent Path Finding in Continuous Time**, which introduces **Optimal Continuous-time Conflict-Based Search (OC-CBS)**.
OC-CBS is based on Continuous-time Conflict-Based Search (CCBS) but with a new branching rule (δ-BR) that restores guarantees of exactness and solution completeness. 
That is, OC-CBS is guaranteed to terminate on any solvable continuous-time MAPF problem with an optimal solution. 
It is currently under review at Artificial Intelligence (Elsevier), the pre-print can be found at [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-B31B1B.svg)](https://www.arxiv.org/abs/2508.16410).

<br> 

## Abstract

_Continuous-time Conflict Based Search (CCBS) has been widely used as an exact baseline for Continuous-time Multi-Agent Path Finding (MAPF<sub>R</sub>),
and its correctness guarantees underpin a range of continuation methods built on top of it.
Recent work, however, has shown that CCBS's guarantees of exactness and solution completeness do not in fact hold:
optimal solutions can be removed from the search, causing the algorithm to return suboptimal solutions. 
This paper establishes
sufficient conditions for exactness and solution completeness in CCBS-style algorithms,
and introduces Optimal Continuous-time Conflict-Based Search (OC-CBS) which satisfies these conditions.
OC-CBS therefore guarantees an optimal solution on every solvable MAPF<sub>R</sub> instance.
Experiments on benchmark problems 
show that OC-CBS remains competitive with CCBS in runtime while providing formal correctness guarantees.
Because OC-CBS is a drop-in replacement for CCBS, it also restores the theoretical guarantees of existing methods that relied on CCBS's now-invalidated correctness.
Finally, the framework and correctness criteria offer a general foundation for analyzing and designing future exact MAPF<sub>R</sub> solvers._ 


<br> 


## Repository Structure

This repository is forked from `PathPlanning/Continuous-CBS:master` and contains two branches:
- ```master```: contains OC-CBS.
- ```originalCCBS```: contains CCBS.

Contents
* [BenchmarkResults](https://github.com/Adcombrink/S-and-SC-CCBS/tree/master/BenchmarkResults) - Benchmarking result files.
* [Counterexample](https://github.com/Adcombrink/S-and-SC-CCBS/tree/master/Counterexample) - Files related to the counter example introduced in the article. 
* [Examples](https://github.com/Adcombrink/S-and-SC-CCBS/tree/master/Examples) - Example problem instance files.
* [Instances](https://github.com/Adcombrink/S-and-SC-CCBS/tree/master/Instances) - Instance files, containing all benchmark problems.
* [LICENSE](https://github.com/Adcombrink/S-and-SC-CCBS/blob/master/LICENSE.md) - License information.
* Remaining files - Source code.

<br>

## Getting Started

Compilation using [CMake](https://cmake.org/) is tested with the provided `CMakeLists`, however, other compilers are available. This project uses C++11 standard. Make sure that your compiler supports it.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installing

Download the current repository to your local machine. Use
```
git clone https://github.com/Adcombrink/Optimal-Continuous-CBS.git
```
or download it directly, then built the CCBS program using, e.g., CMake:
```bash
cd PATH_TO_THE_PROJECT
cmake .
make CCBS
```

### Solving a single MAPF problem
The solvers are launched with XML file input arguments (see [Examples](https://github.com/Adcombrink/S-and-SC-CCBS/tree/master/Examples)):
- `map`: an XML file containing the map data. Two map structures are used: gridmaps and roadmaps. 
- `tasks`: an XML file containing the start/goal vertex pairs for each agent.
- `config` (optional): an XML file containing parameter values (see below). Default values are used if no config file is given.

For example:
```
./CCBS map.xml task.xml config.xml
```
runs OC-CBS/CCBS on the given problem and outputs a result file in the same directory named as the task-file + `_log.xml`.

#### Config options

* `<use_cardinal>` - controls whether the algorithm is looking for cardinal and semi-cardinal collisions or not. Possible values are `1`(true) or `0` (false).
* `<use_disjoint_splitting>` - From the original CCBS repository. This is currently **not supported** with δ-BR.
* `<hlh_type>` - From the original CCBS repository. This is currently **not supported** with δ-BR.
* `<connectedness>` - controls the connectedness of the grid. Possible values: `2` - 4 cardinal neighbors; `3` - 4 cardinal + 4 diagonal; `4` - 16 neighbors; `5` - 32 neighbors. In case if the map is represented as roadmap this parameter is ignored.
* `<timelimit>` - controls the maximum runtime of the algorithm. Possible values are >0. For example value 60 means that the algorithm can spend up to 60 seconds to find a solution.
* `<agent_size>` - controls the size (radii) of the agents' shape. Possible values are in the range (0, 0.5].
* `<precision>` - controls how precise the end of collision interval is detected (the moment of time when there is no more collision between the agents). The lower the value - the preciser the algorithm finds the end of collision interval, but it takes a bit more time. Possible values are >0.
* `<branching_gamma>` - controls the gamma value used in δ-BR. This option is not available on the ```originalCCBS``` branch.

### Running a benchmark set

To evaluate the solver across a full instance set (e.g. one of the [MovingAI MAPF benchmarks](https://movingai.com/benchmarks/mapf.html)), two additional tools are provided: `movingai_converter`, which converts MovingAI `.map`/`.scen` files into the `map.xml`/task-xml format used by this project, and `run_benchmarks.sh`, which sweeps CCBS over an increasing number of agents for every scenario.

Build the converter alongside the solver:
```bash
cmake .
make CCBS movingai_converter
```

Lay out one subfolder per instance set under a main folder, each containing a MovingAI map file and its scenario files, e.g.:
```
Instances/
  room-32-32-4/
    room-32-32-4.map
    room-32-32-4-random-1.scen
    room-32-32-4-random-2.scen
    ...
```
(`movingai_converter` requires exactly one `.map` file and at least one `.scen` file per subfolder; a subfolder that already contains a converted `map.xml` is used as-is.)

Then run:
```bash
./run_benchmarks.sh Instances --kmin 2 --kmax 4 --jobs 8
```
This converts every subfolder in `Instances` that isn't already converted, then, for each subfolder, each connectedness value from `2` to `4`, and each scenario, runs CCBS starting at 2 agents and incrementing by one (taking the first N agents from the scenario, in file order) until a run is unsolved or times out. Up to 8 of these `(subfolder, connectedness, scenario)` sweeps run in parallel, so scenario files within the same connectedness value are processed concurrently. Results are aggregated into `Instances/benchmark_results.csv` (one row per run: agent count, connectedness, solved, runtime, makespan, flowtime, and search-tree statistics).

Full usage:
```
./run_benchmarks.sh <main_folder> [options]
```
`main_folder` is the only required, positional argument; everything else is an optional flag (`--flag value` or `--flag=value`):
- `--kmin`, `--kmax` - connectedness range to sweep (2-5, inclusive; default: 2-5).
- `--max-agents` - cap on the agent-count sweep per scenario (default: no cap).
- `--output` - where to write the aggregated CSV (default: `<main_folder>/benchmark_results.csv`).
- `--jobs` - number of `(subfolder, connectedness, scenario)` sweeps to run in parallel (default: 1). Each agent-count sweep within one scenario stays sequential, since it stops at the first unsolved size.
- `--config` - config.xml template to sweep from (see [Config options](#config-options) above; default: `Examples/config.xml`). `--kmin`/`--kmax` override its `<connectedness>` per run; every other field is taken from the template as-is. There is no `--timelimit` flag: the per-run time limit comes only from the template's `<timelimit>`, which also sizes the external watchdog process that enforces it (falls back to CCBS's own built-in default, 30s, if the tag is missing).


