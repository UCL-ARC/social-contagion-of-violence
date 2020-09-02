# README #

### What is this repository for? ###

Python-based framework to simulate, analyse and infer infections in a network assuming a Hawkes contagion process and accounting for various confounding effects.

### Version: 0.1

### Instructions ###

* Summary of set up: Run main.py to generate simulations and plots as is
* Configuration:
    - To simulate different regimes modify input/simulationparams.py
	- To modify experimental setup parameters such as coverage modify main.py
    - If running from the command line follow instructions provided in main.py to ensure plots are not displayed
* Dependencies: ```conda env create -f environment.yml```
* How to run tests: ```pytest tests/*```
* Deployment instructions: Not applicable
