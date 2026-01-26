.PHONY: clean simulation

clean:
	rm -r results*

simulation_set_1: run.py
	OMP_NUM_THREADS=1 python run.py 1

simulation_set_2: run.py
	OMP_NUM_THREADS=1 python run.py 2

simulation_set_3: run.py
	OMP_NUM_THREADS=1 python run.py 3

simulation_set_4: run.py
	OMP_NUM_THREADS=1 python run.py 4

simulation_set_5: run.py
	OMP_NUM_THREADS=1 python run.py 5

simulations: simulation_set_1 simulation_set_2 simulation_set_3 simulation_set_4 simulation_set_5

display_results: display_results.py
	python display_results.py

