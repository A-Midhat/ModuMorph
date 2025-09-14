Purpose
- Short, actionable guidance for an AI coding agent working on ModuMorph (universal morphology control).

Quick architecture summary
- This repo is a fork/extension of MetaMorph: transformer-based universal policies + hypernetworks.
- Major components:
  - configs/: YAML experiment configs (e.g. `configs/mr_mt_runs.yaml`) drive model, env, and PPO settings.
  - tools/: training/eval utilities (`tools/train_ppo.py`, `tools/zs_morph_eval.py`, analysis scripts).
  - metamorph/: core library (models, env wrappers) — treat this as the project runtime API.
  - artifacts/: training outputs. Each run dir contains `checkpoint_*.pt` and `config.yaml` used at eval time.

Developer workflows (essential commands)
- Build docker: `./scripts/build_docker.sh` (recommended for reproducibility).
- Train (example):
  python tools/train_ppo.py --cfg configs/mr_mt_runs.yaml OUT_DIR ./output/<name> RNG_SEED 1409
- Zero-shot eval (example):
  python tools/zs_morph_eval.py --run_dir artifacts/<run_dir> --checkpoint checkpoint_600.pt \
    --morph Panda --task Door --controller OSC_POSE --episodes 5 --save_video ./test_videos/ --debug
- Aggregate results: scripts under `tools/` (e.g. `tools/aggregate_morph_stats.py`) read `results_by_type/*/aggregated.csv`.

Project-specific conventions & patterns
- Config-driven: almost all behavior is controlled by `cfg.merge_from_file(config.yaml)` and CLI overrides. When editing behavior, prefer adding/overriding YAML keys instead of hardcoding.
- Training vs evaluation:
  - Training configs use `PPO.NUM_ENVS` (often 32). Evaluation forces `PPO.NUM_ENVS=1` and may toggle `ENV_ARGS.has_offscreen_renderer`.
  - Artifacts contain a `config.yaml` snapshot; evaluation scripts rely on that to recreate envs.
- Morph/task indexing: config lists `ROBOSUITE.TRAINING_MORPHOLOGIES` and `ROBOSUITE.ENV_NAMES`. Generalization to unseen morphs is handled by temporarily extending these lists (see `tools/zs_morph_eval.py`).
- Object-node options: model config keys like `MODEL.ADD_OBJECT_NODE`, `OBJECT_POSE_IN_CONTEXT`, and `MODEL.TRANSFORMER.FILM_CONTEXT_MODE` change how object observations are included. Search these keys when changing observation processing.
- Task embedding vs pair-indexing: models may contain `v_net.task_embed`. `zs_morph_eval.py` prefers task-embedding indexing for generalization; change here only with care.

Integration points & external deps
- Robosuite (task envs) — env creation via `metamorph.algos.ppo.envs.make_vec_envs`.
- MuJoCo/Unimal-100 assets required (follow MetaMorph README referenced in project README).
- Logging: Weights & Biases (wandb) is used when `LOGGING.USE_WANDB=True`.
- Checkpoints: PyTorch `torch.save`/`torch.load` for `checkpoint_*.pt`. Eval expects either tuple (actor_critic, ob_rms) or dict with `actor_critic` key.

Where to make targeted changes
- Add/modify tasks or morph lists: update YAML (configs/*.yaml) and ensure `ROBOSUITE.ENV_ARGS` settings remain consistent.
- Change model topology or decoder dims: modify `MODEL.TRANSFORMER.*` fields in configs and corresponding model code under `metamorph/`.
- Add new eval metrics or export formats: update `tools/zs_morph_eval.py` (it has `--save_metrics` and `--out_dir` hooks).

Quick coding guidelines for AI agents
- Prefer config changes over code changes when possible; provide a YAML snippet example.
- When editing model code, include a small unit test that instantiates the module and runs a forward pass with a minimal fake observation (use `PPO.NUM_ENVS=1` and small dims).
- If you change env-observation layout, update `ENV.KEYS_TO_KEEP` in configs and search for uses of those keys (`'object-state'`, `'proprioceptive'`, `'edges'`, ...).
- Preserve artifact format: do not change checkpoint file layout unless you also update `tools/zs_morph_eval.py` loader logic.

Helpful file pointers (examples)
- `configs/mr_mt_runs.yaml` — canonical MR-MT training config (tasks, morphs, transformer options).
- `tools/zs_morph_eval.py` — zero-shot evaluation flow, generalization logic, and examples of loading checkpoints and creating envs.
- `tools/train_ppo.py` — training entrypoint; accepts CLI YAML overrides.
- `artifacts/<run>/config.yaml` and `checkpoint_*.pt` — evaluation looks for these in run dirs.

If unsure, ask the user for
- preferred GPU / Docker vs local venv for running heavy jobs
- whether new features should keep backward-compatibility with existing checkpoints

End of instructions.
