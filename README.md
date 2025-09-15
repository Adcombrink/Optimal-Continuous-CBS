
<br>
<h1 align="center">Optimal Multi-Agent Path Finding in Continuous Time</h1>
<br>
  <p align="center">
    Alvin Combrink, Sabino Franceso Roselli, and Martin Fabian.
  </p>
  <br>
<br>

This is the official respository for **Optimal Multi-Agent Path Finding in Continuous Time**, which introduces a new branching rule ($\delta$-BR) for CCBS with soundness and solution-completeness guarantees. That is, CCBS using this branching rule (CCBS- $\delta$-BR) is guaranteed to terminate on any solvable continuous-time MAPF problem with an optimal solution. The pre-print version is available at [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-B31B1B.svg)](https://www.arxiv.org/abs/2508.16410).

<br> 

## Abstract
_Continuous-time Conflict Based-Search (CCBS) has long been viewed as the de facto optimal solver for multi-agent path finding in continuous time (MAPFR), yet recent critiques show that the theoretically described CCBS can fail to terminate on solvable MAPFR problems while the publicly available reference implementation can return sub-optimal solutions. This work presents an analytical framework that yields simple and sufficient conditions under which any CCBS-style algorithm is both sound (returns only optimal solutions) and solution complete (terminates on every solvable MAPFR problem). Investigating the reference implementation reveals that it violates the soundness conditions, with counterexamples demonstrating sub-optimality._

_Leveraging the framework, we introduce a branching rule (δ-BR) and prove it restores soundness and termination guarantees. Consequently, the resulting CCBS variant is both sound and solution complete, matching the guarantees of the discrete-time CBS for the first time in the continuous domain. On a constructed example, CCBS with δ-BR improves sum-of-costs from 10.707 to 9.000 (≈16 % lower) compared to the reference implementation. Across benchmarks, the reference implementation is generally able to find solutions faster than CCBS with δ-BR due to its more aggressive pruning. However, this comes at the cost of occasional sub-optimality and potential non-termination when all solutions are pruned, whereas δ-BR preserves optimality and guarantees termination by design. Because δ-BR largely only affects the branching step, it can be adopted as a drop-in replacement in existing codebases, as we show in our provided implementation. Beyond CCBS, the analytical framework and termination criterion provide a systematic way to evaluate other CCBS-like MAPFR solvers and future extensions._

<br> 


## Repository Structure

This repository is forked from `PathPlanning/Continuous-CBS:master` and contains two branches:
- ```master```: contains CCBS using δ-BR.
- ```originalCCBS```: contains CCBS using the original branching rule.

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

Download current repository to your local machine. Use
```
git clone https://github.com/Adcombrink/S-and-SC-CCBS.git
```
or download it directly, then built the CCBS program using, e.g., CMake:
```bash
cd PATH_TO_THE_PROJECT
cmake .
make CCBS
```

### CCBS on a single MAPF problem
CCBS is launched with XML file input arguments (see [Examples](https://github.com/Adcombrink/S-and-SC-CCBS/tree/master/Examples)):
- `map`: an XML file containing the map data. Two map structures are used: gridmaps and roadmaps. 
- `tasks`: an XML file containing the start/goal vertex pairs for each agent.
- `config` (optional): an XML file containing parameter values (see below). Default values are used if no config file is given.

For example:
```
./CCBS map.xml task.xml config.xml
```
runs CCBS on the given problem and outputs a result file in the same directory named as the task-file plus `_log.xml`.

#### Config options

* `<use_cardinal>` - controls whether the algorithm is looking for cardinal and semi-cardinal collisions or not. Possible values are `1`(true) or `0` (false).
* `<use_disjoint_splitting>` - From the original CCBS repository. This is currently **not supported** with δ-BR.
* `<hlh_type>` - From the original CCBS repository. This is currently **not supported** with δ-BR.
* `<connectedness>` - controls the connectedness of the grid. Possible values: `2` - 4 cardinal neighbors; `3` - 4 cardinal + 4 diagonal; `4` - 16 neighbors; `5` - 32 neighbors. In case if the map is represented as roadmap this parameter is ignored.
* `<timelimit>` - controls the maximum runtime of the algorithm. Possible values are >0. For example value 60 means that the algorithm can spend up to 60 seconds to find a solution.
* `<agent_size>` - controls the size (radii) of the agents' shape. Possible values are in the range (0, 0.5].
* `<precision>` - controls how precise the end of collision interval is detected (the moment of time when there is no more collision between the agents). The lower the value - the preciser the algorithm finds the end of collision interval, but it takes a bit more time. Possible values are >0.
* `<branching_gamma>` - controls the gamma value used in δ-BR. This option is not available on the ```originalCCBS``` branch.



### Benchmarking CCBS
An additional program `benchmarking` is provided to benchmark CCBS on a map and multiple task files according to the benchmarking scheme detailed in the article. To build:
```bash
make benchmarking
```
Run using
```
./benchmarking folder kmin kmax num_processes
```
where 
- `folder`: a folder containing one map XML file named `map.xml`, one config file named 'config.xml' and a number of task XML files.
- `kmin` and `kmax`: a minimum and maximum connectedness value. This is only used for gridmaps. Every integer from `kmin` to `kmax` will be run.
- `num_processes`: the maximum number of processes to spawn, for multiprocessing instances on separate CPU cores.

A result file will be output in `folder`.





