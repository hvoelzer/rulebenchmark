# rulebenchmark
Towards Comprehensive Benchmarking for Rule Induction in Transparent Automated Business Decisions

## Inspect full results of run 1
- go to https://github.com/hvoelzer/rulebenchmark/blob/main/rulebenchmark/viz-svg.ipynb
- download the raw results at https://github.com/hvoelzer/rulebenchmark/blob/main/rulebenchmark/results_1.csv


## Inspect results of other runs, change charts
- use and modify viz.ipynb
- you can use any of the results_k.csv files in viz.ipynb


## Re-run the benchmarking on your hardware
- install this project, run benchmarking with pre-configured test data
```
virtualenv venv
source venv/bin/activate
git clone git@github.com:hvoelzer/rulebenchmark.git
cd rulebenchmark
pip install -e .
cd rulebenchmark
python run_benchmarking.py
```
- download missing data sets into data directory (links to sources in data_configs.py)
- configure main parameters in run_benchmarking.py
- execute run_benchmarking.py


## Add a data set, or run only selected data sets
- put a new csv file into data directory
- add a data set configuration in data_configs.py
- specify in run_benchmarking.py which data sets to include in a run

## Add a rule induction pipeline, or select pipelines to run
- integrate a new pipeline into run_benchmarking.py by configuring each step of the pipeline

