"""The tuning workflow: the objective, surviving Optuna storage errors, run.py hyperparameter flags,
tune.py and validate.py."""

import json

import pytest

torch = pytest.importorskip("torch")
optuna = pytest.importorskip("optuna")

from agents.ppo import PPOConfig, Run, load_checkpoint, load_metrics, load_runs, train  # noqa: E402
from tests.test_tracking import tiny_cfg  # noqa: E402


def sqlite(tmp_path, name="optuna.db"):
    return f"sqlite:///{tmp_path.as_posix()}/{name}"


def tune_study(tmp_path, values=(0.6, 0.8)):
    """A ppo-tune study in tmp_path/optuna.db with COMPLETE trials carrying the tuned params, plus a
    fixed n_updates like record_config adds (which --from-trial must not copy)."""
    from agents.ppo.tune import TUNED_PARAMS

    storage = sqlite(tmp_path)
    study = optuna.create_study(study_name="ppo-tune", storage=storage, direction="maximize")
    dists = {
        "lr": optuna.distributions.FloatDistribution(1e-5, 1e-3, log=True),
        "entropy_coef": optuna.distributions.FloatDistribution(0.0, 0.05),
        "clip_eps": optuna.distributions.FloatDistribution(0.1, 0.3),
        "gae_lambda": optuna.distributions.FloatDistribution(0.9, 0.99),
        "n_epochs": optuna.distributions.IntDistribution(2, 8),
        "n_updates": optuna.distributions.IntDistribution(200, 200),
    }
    assert set(TUNED_PARAMS) <= set(dists)
    for i, v in enumerate(values):
        params = {"lr": 1e-4 * (i + 1), "entropy_coef": 0.02, "clip_eps": 0.15, "gae_lambda": 0.93,
                  "n_epochs": 2 + i, "n_updates": 200}
        study.add_trial(optuna.trial.create_trial(params=params, distributions=dists, value=v))
    return storage


# --- the objective ---------------------------------------------------------------------------

def test_objective_is_eval_battle_win_rate_and_best_pt_follows_it(tmp_path):
    from agents.ppo.train import OBJECTIVE

    assert OBJECTIVE == "eval/battle_win_rate"
    cfg = tiny_cfg(tmp_path, n_updates=4, eval_every=1, checkpoint_every=0)
    result = train(cfg, "obj", log_fn=None)
    m = load_metrics(result.run_dir)
    curve = m[OBJECTIVE].dropna()
    assert len(curve) == 4 and "eval/win_rate" in m.columns  # win_rate is still logged
    assert result.best_objective == pytest.approx(curve.max())
    row = load_runs(runs_dir=str(tmp_path)).iloc[-1]
    assert row["value"] == pytest.approx(curve.max()) and row["user_attrs_objective"] == OBJECTIVE
    first_best = int(curve.idxmax())  # best.pt is saved on strictly greater: the first maximum
    _, _, best = load_checkpoint(result.run_dir / "checkpoints" / "best.pt", device="cpu")
    assert best["update"] == first_best and best[OBJECTIVE] == pytest.approx(curve.max())
    assert row["user_attrs_best_update"] == first_best


def test_default_eval_episodes_is_large_enough_to_rank_trials():
    assert PPOConfig().eval_episodes >= 50


# --- tracking failures never stop training ----------------------------------------------------

def test_deleting_the_study_mid_run_does_not_stop_training(tmp_path, caplog):
    """The update-99 crash: the study was deleted in optuna-dashboard while the run wrote to it."""
    from agents.ppo.experiments import default_storage

    lines = []

    def log_fn(line):
        lines.append(line)
        if len(lines) == 2:  # right after update 0 (the first line is the run header)
            optuna.delete_study(study_name="ppo", storage=default_storage(tmp_path))

    cfg = tiny_cfg(tmp_path, n_updates=4, eval_every=1, checkpoint_every=0)
    with caplog.at_level("WARNING", logger="agents.ppo.experiments"):
        result = train(cfg, "deleted", log_fn=log_fn)
    assert list(load_metrics(result.run_dir).index) == [0, 1, 2, 3]
    _, _, latest = load_checkpoint(result.run_dir / "checkpoints" / "latest.pt", device="cpu")
    assert latest["update"] == 3
    assert (result.run_dir / "checkpoints" / "best.pt").exists() and result.best_objective is not None
    assert "training continues" in caplog.text


def test_storage_guard_never_swallows_pruning(tmp_path):
    storage = sqlite(tmp_path, "t.db")
    study = optuna.create_study(storage=storage, pruner=optuna.pruners.ThresholdPruner(lower=0.5))
    with pytest.raises(optuna.TrialPruned):
        with Run(tiny_cfg(tmp_path), "pruned", storage=storage, trial=study.ask(), tensorboard=False) as run:
            run.report(0, 0.1, prune=True)


def test_interrupted_update_is_redone_on_resume(tmp_path, monkeypatch):
    """latest.pt records the last *finished* update, not the one cut off mid-rollout."""
    import importlib

    train_mod = importlib.import_module("agents.ppo.train")  # the package re-exports `train`, the function
    real, calls = train_mod.collect_rollout, []

    def flaky(*a, **kw):
        calls.append(1)
        if len(calls) == 3:  # during update 2's rollout
            raise KeyboardInterrupt("simulated SIGTERM")
        return real(*a, **kw)

    cfg = tiny_cfg(tmp_path, n_updates=4, eval_every=0, checkpoint_every=0)
    monkeypatch.setattr(train_mod, "collect_rollout", flaky)
    with pytest.raises(KeyboardInterrupt):
        train(cfg, "cut", log_fn=None)
    _, _, latest = load_checkpoint(tmp_path / "cut" / "checkpoints" / "latest.pt", device="cpu")
    assert latest["update"] == 1
    monkeypatch.setattr(train_mod, "collect_rollout", real)
    result = train(cfg, "cut", log_fn=None, resume=True)
    assert list(load_metrics(result.run_dir).index) == [0, 1, 2, 3]  # update 2 is not skipped


# --- run.py hyperparameter flags ---------------------------------------------------------------

def test_run_flags_reach_the_config(tmp_path):
    from agents.ppo.run import build_config, parser

    args = parser().parse_args([
        "--lr", "3e-4", "--entropy-coef", "0.03", "--clip-eps", "0.25", "--gae-lambda", "0.9",
        "--n-epochs", "6", "--eval-episodes", "7", "--updates", "10", "--runs-dir", str(tmp_path),
    ])
    cfg = build_config(args)
    got = (cfg.lr, cfg.entropy_coef, cfg.clip_eps, cfg.gae_lambda, cfg.n_epochs, cfg.eval_episodes, cfg.n_updates)
    assert got == (3e-4, 0.03, 0.25, 0.9, 6, 7, 10)
    assert build_config(parser().parse_args([])) == PPOConfig()  # no flags: the defaults


def test_from_trial_and_params_json_with_explicit_flags_winning(tmp_path):
    from agents.ppo.run import build_config, parser

    tune_study(tmp_path)
    (tmp_path / "p.json").write_text(json.dumps({"clip_eps": 0.11, "eval_every": 5}))
    args = parser().parse_args([
        "--runs-dir", str(tmp_path), "--from-trial", "ppo-tune:1", "--params-json", str(tmp_path / "p.json"),
        "--n-epochs", "8",
    ])
    cfg = build_config(args)
    assert cfg.lr == pytest.approx(2e-4) and cfg.entropy_coef == pytest.approx(0.02)  # from the trial
    assert cfg.n_updates == PPOConfig.n_updates  # the trial's fixed n_updates=200 is not copied
    assert cfg.clip_eps == 0.11 and cfg.eval_every == 5  # --params-json beats the trial
    assert cfg.n_epochs == 8                              # an explicit flag beats both
    assert build_config(parser().parse_args(["--params-json", '{"hidden": [16]}'])).hidden == (16,)
    for bad in (["--from-trial", "ppo-tune:9"], ["--from-trial", "nope:0"], ["--from-trial", "ppo-tune"],
                ["--params-json", '{"not_a_field": 1}']):
        with pytest.raises(SystemExit):
            build_config(parser().parse_args(["--runs-dir", str(tmp_path), *bad]))


def test_run_main_end_to_end_with_tuned_params(tmp_path, monkeypatch, capsys):
    import agents.ppo.run as run_mod

    monkeypatch.setattr(run_mod.signal, "signal", lambda *a: None)  # leave pytest's SIGTERM alone
    tune_study(tmp_path)
    tiny = json.dumps({"hidden": [16], "async_envs": False, "n_steps": 16, "minibatch_size": 16})
    argv = ["--name", "final", "--runs-dir", str(tmp_path), "--updates", "2", "--n-envs", "2", "--eval-every", "1",
            "--eval-episodes", "1", "--params-json", tiny, "--from-trial", "ppo-tune:0", "--resume"]
    run_mod.main(argv)
    saved = json.loads((tmp_path / "final" / "config.json").read_text())
    assert saved["lr"] == pytest.approx(1e-4) and saved["n_epochs"] == 2 and saved["eval_episodes"] == 1
    runs = load_runs(runs_dir=str(tmp_path))
    assert runs.iloc[-1]["user_attrs_params_from_trial"] == "ppo-tune:0"
    run_mod.main(argv)  # what the container does after the run finished
    assert "nothing to do" in capsys.readouterr().out


# --- tune.py ---------------------------------------------------------------------------------

def test_tune_keeps_going_after_a_crashed_trial():
    from agents.ppo.tune import run_trials

    study = optuna.create_study(direction="maximize")

    def objective(trial):
        if trial.number == 0:
            raise RuntimeError("boom")
        return 0.5

    run_trials(study, objective, 3)
    S = optuna.trial.TrialState
    assert [t.state for t in study.trials] == [S.FAIL, S.COMPLETE, S.COMPLETE]


def test_tune_keeps_going_when_optuna_itself_fails():
    from agents.ppo.tune import run_trials

    study = optuna.create_study(direction="maximize")
    real_optimize, calls = study.optimize, []

    def optimize(*a, **kw):  # an error raised by Optuna's own tell, outside `catch`
        calls.append(1)
        if len(calls) == 1:
            raise KeyError("Record does not exist.")
        return real_optimize(*a, **kw)

    study.optimize = optimize
    run_trials(study, lambda t: 0.5, 2)
    assert len(calls) == 2 and len(study.trials) == 1


def test_tune_cli_flags_reach_the_trials(tmp_path, monkeypatch):
    import agents.ppo.tune as tune_mod

    class Result:
        best_objective = 0.5

    seen = []
    monkeypatch.setattr(tune_mod, "train", lambda cfg, name, **kw: seen.append(cfg) or Result())
    monkeypatch.setattr(tune_mod.signal, "signal", lambda *a: None)
    tune_mod.main(["--trials", "2", "--updates", "7", "--n-envs", "3", "--eval-episodes", "4",
                   "--storage", sqlite(tmp_path)])
    assert len(seen) == 2 and all((c.n_updates, c.n_envs, c.eval_episodes) == (7, 3, 4) for c in seen)
    # rerunning the same command continues the study
    tune_mod.main(["--trials", "1", "--updates", "7", "--storage", sqlite(tmp_path)])
    assert len(optuna.load_study(study_name="ppo-tune", storage=sqlite(tmp_path)).trials) == 3


# --- validate.py -----------------------------------------------------------------------------

def test_validate_reruns_top_trials_with_seeds_and_resumes(tmp_path):
    from agents.ppo.validate import validate

    storage = tune_study(tmp_path, values=(0.6, 0.8, 0.7))
    base = tiny_cfg(tmp_path, n_updates=2, eval_every=1, checkpoint_every=0)
    lines = []
    table = validate("ppo-tune", storage, 1, [1, 2], base, log_fn=lines.append)
    assert [r["source"] for r in table] == ["ppo-tune:1"]  # the best trial (0.8)
    assert table[0]["seeds_done"] == 2
    for s in (1, 2):
        cfg = json.loads((tmp_path / f"val-ppo-tune-t001-s{s}" / "config.json").read_text())
        assert cfg["seed"] == s and cfg["lr"] == pytest.approx(2e-4) and cfg["n_updates"] == 2
    assert any("--from-trial ppo-tune:1" in line for line in lines)
    assert set(load_runs(study_name="ppo-validate", runs_dir=str(tmp_path))["state"]) == {"COMPLETE"}
    lines.clear()
    validate("ppo-tune", storage, 1, [1, 2], base, log_fn=lines.append)
    assert sum(line.endswith(": done") for line in lines) == 2  # finished runs are skipped
