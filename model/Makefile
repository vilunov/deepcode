PYTHONPATH = PYTHONPATH=./
TEST = $(PYTHONPATH) pytest --verbosity=2 --showlocals --strict $(arg) -k "$(k)"

.PHONY: test

test:
	$(TEST) --cov=deepcode
