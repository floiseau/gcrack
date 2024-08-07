.PHONY: clean simulation display_results

clean:
	rm -r results*

simulation: run.py
	OMP_NUM_THREADS=1 python run.py

display_results: display_results.py
	python display_results.py

