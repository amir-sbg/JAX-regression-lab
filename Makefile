.PHONY: install run test clean

install:
	python -m pip install -e .

run:
	python -m jax_regression.pipeline

test:
	python -m pytest -q

clean:
	rm -rf artifacts reports .pytest_cache src/jax_regression_lab.egg-info
