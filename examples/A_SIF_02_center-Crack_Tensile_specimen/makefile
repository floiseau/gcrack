.PHONY: clean simulation

clean:
	rm -r results* *.csv *.pdf

simulation: run.py
	OMP_NUM_THREADS=1 python run.py
