.PHONY: clean simulation

clean:
	rm -r results*

simulation: run.py
	OMP_NUM_THREADS=1 python -u run.py

plots:
	python display_results.py
