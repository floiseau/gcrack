.PHONY: clean simulations simulation_alpha_15 simulation_alpha_30 simulation_alpha_45 simulation_alpha_60 simulation_alpha_75 display_comparison_60 display_comparison_all

clean:
	rm -r results*

simulation_alpha_15: run.py
	OMP_NUM_THREADS=1 python run.py 15

simulation_alpha_30: run.py
	OMP_NUM_THREADS=1 python run.py 30

simulation_alpha_45: run.py
	OMP_NUM_THREADS=1 python run.py 45

simulation_alpha_60: run.py
	OMP_NUM_THREADS=1 python run.py 60

simulation_alpha_75: run.py
	OMP_NUM_THREADS=1 python run.py 75

simulations: simulation_alpha_15 simulation_alpha_30 simulation_alpha_45 simulation_alpha_60 simulation_alpha_75

display_comparison_60: display_comparison_60.py
	python display_comparison_60.py

display_comparison_all: display_comparison_all.py
	python display_comparison_all.py

