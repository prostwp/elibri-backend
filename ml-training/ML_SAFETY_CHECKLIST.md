# ML Pipeline Safety Checklist

> После 9 раундов ревью и 45 багов в Сценарии A — этот чек-лист обязателен
> перед написанием кода для сценариев B-K.

## 1. Прежде чем писать НОВУЮ функцию

- [ ] Написать 2-3 unit-теста в `tests/` ДО реализации (TDD)
- [ ] Зафиксировать ожидаемые invariants (что должно быть истиной)
- [ ] Аннотировать типы аргументов и возврата

## 2. Прежде чем писать ML-функцию

### Look-ahead audit
- [ ] Каждое `pd.Series.rolling/.corr/.rank/.mean` — есть `.shift(1)` если использует current bar?
- [ ] `np.roll(x, 1)` — index 0 затёрт нулём?
- [ ] Cross-asset features — timestamp aligned via `merge_asof`?
- [ ] Regime label — стрикто-past info?

### Label leakage audit
- [ ] Любой train/test split — горизонт `purge gap` отнят?
- [ ] Inner AG `tuning_data` — chronological + purge gap?
- [ ] `num_bag_folds=0` (запрет AG random k-fold)?

### Sample weights (для overlapping labels)
- [ ] `compute_sample_uniqueness` вызвана если horizon > 1?
- [ ] `_sample_weight` колонка проброшена через build_dataset → train_predictor / train_specialist?
- [ ] AG fit получает `sample_weight="_sample_weight"`?

### Sharpe / metric math
- [ ] Position-return alignment: `pos[t] * ret_next[t]` (не `ret[t]`)?
- [ ] Fees only on transitions: `pos_change = np.abs(pos - pos_prev)`?
- [ ] MIN_TRADES_FOR_SHARPE guard?
- [ ] BARS_PER_YEAR right (365 для крипты, не 252)?

### JSON / IO
- [ ] `atomic_write_json` (sanitize NaN + tmp+rename)?
- [ ] OHLCV validation in fetch_or_cache?
- [ ] Symbol regex validation в CLI?

### AG specifics
- [ ] `ag_args_fit={"num_gpus": 1}` (не `hyperparameters=`)?
- [ ] `tuning_data=` chronological (не random)?
- [ ] `class_labels = [int(x) for x in pred.class_labels]` для AG version drift?
- [ ] Feature drift warn в evaluate_predictor?

## 3. Перед запуском training

- [ ] `make check` (pytest + mypy) проходит
- [ ] `make smoke` (build_dataset на 2 годах) — sane class distribution
- [ ] rsync только нужных файлов на vast.ai (не вся ml-training/)
- [ ] tmux session для resilience

## 4. После training

- [ ] Verdict записан в JSON (atomic write)
- [ ] HANDOFF.md SESSION LOG обновлён
- [ ] DEVLOG.md секция per-patch добавлена
- [ ] Commit с co-author

## Anti-patterns которые НЕ делать

- ❌ `from train import TF_CONFIG` (legacy) — используй train_autogluon
- ❌ `np.random.*` без seed
- ❌ `json.dump(...)` без `default=str` или sanitize
- ❌ `df.to_parquet(path)` без tmp+replace
- ❌ Триплбарьер с tb_upper != tb_lower (структурный bias)
- ❌ Mock-данные в production (np.random.normal с seed=42)
- ❌ Запуск ML на маке (только vast.ai)
- ❌ Писать >500 LOC новых файлов БЕЗ тестов первого порядка

## Bugs found in Scenario A (для прецедента)

См `~/.claude/projects/-Users-admin-NodeElibiri/memory/feedback_ml_pipeline_safety.md` —
полный список 45 багов сгруппированных по 8 классам. Каждый класс имеет
regression test в `tests/test_metrics.py`.
