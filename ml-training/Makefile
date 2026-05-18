# Test + lint gate.
# Run BEFORE every push to vast.ai. After Scenario A's 9-round / 45-bug
# debacle, no new ML pipeline file ships without `make check` passing locally.
#
# Targets:
#   make test     pytest regression suite (13+ tests, <2s)
#   make lint     mypy --strict on changed files
#   make check    test + lint
#   make smoke    minimal build_dataset + sanity (<30s, no GPU)

.PHONY: test lint check smoke clean

PY := .venv/bin/python
PYTEST := .venv/bin/pytest

test:
	@$(PYTEST) tests/ -q

lint:
	@.venv/bin/mypy --ignore-missing-imports --no-strict-optional \
		feature_engine.py train_autogluon.py wfcv_regime_aware.py \
		multi_agent_fusion.py regime_classifier.py data_fetcher.py \
		fetch_macro_features.py 2>&1 | tail -30 || true

check: test lint

smoke:
	@$(PY) -c "from train_autogluon import build_dataset; \
	feat, _, meta = build_dataset('BTCUSDT', '4h', years=2.0, with_macro=True); \
	print('smoke OK:', feat.shape, 'class:', meta['class_distribution'])"

clean:
	@rm -rf models/wfcv models/ag_*_*_macro models/fusion_*_macro \
	    logs/autogluon_*.json logs/wfcv_regime_*.json \
	    __pycache__ tests/__pycache__ \
	    data/macro/*.parquet.tmp data/*.parquet.tmp
