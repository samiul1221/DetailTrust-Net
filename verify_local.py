"""CPU-only engineering verification using explicitly synthetic volumes."""
from pathlib import Path
from dataclasses import replace
import json
import time
import torch
from detailtrust.config import configuration, experiment_registry, write_json, seed_everything
from detailtrust.data import SyntheticStore, make_split
from detailtrust.experiments import run_experiment_sweep, export_best_detailtrust
from detailtrust.reporting import build_report, export_report, verify_artifacts
from detailtrust.checks import run_unit_checks, tiny_overfit_check
from detailtrust.training import profile_training_stages


def main():
    torch.set_num_threads(4)
    root = Path("detailtrust_outputs/synthetic_verification")
    root.mkdir(parents=True, exist_ok=True)
    cfg = configuration("smoke", output_dir=str(root), patch=(16,16,16), precision="float32",
                        synthetic=True, updates_a=1, updates_b=1, updates_c=1,
                        validation_every=1, epoch_updates=1, profile_warmup=0, profile_updates=1,
                        variants=tuple(experiment_registry()), max_hours=1.)
    device = torch.device("cpu")
    store = SyntheticStore(size=16)
    split = make_split([dict(id=s) for s in store.records], cfg.split_seed)
    write_json(root/"split_manifest.json", split)
    write_json(root/"experiment_config.json", vars(cfg))
    checks = run_unit_checks()
    write_json("verification/unit_checks.json", checks)
    overfit = tiny_overfit_check(store, split["train"][:4], cfg, device, updates=10)
    write_json("verification/tiny_overfit_check.json", overfit)
    profiles = profile_training_stages(store, split["train"][:4], cfg, experiment_registry()["B5"], device)
    write_json("verification/cpu_synthetic_profile.json", profiles)
    statuses = run_experiment_sweep(store, split, cfg, device)
    best = export_best_detailtrust(statuses, root, device)
    report, records, summary = build_report(cfg, split, {"status":"synthetic_only"}, statuses, best,
        checks, profiles, {"identifier":"synthetic_execution_fixture", "real_BraTS_loaded":False})
    artifacts = export_report(report, records, summary, root)
    (root/"README.md").write_text("SYNTHETIC CPU ENGINEERING TEST ONLY. No medical result.\n")
    verify_artifacts(root)
    write_json("verification/integration_status.json", {"evidence":"CPU synthetic execution only",
        "statuses":{k:{"status":v["status"], "reason":v.get("reason")} for k,v in statuses.items()},
        "best_checkpoint_strict_reload":best["status"], "report_parsed":True,
        "records":len(records["rows"]), "artifacts":artifacts})
    print({key: value["status"] for key,value in statuses.items()})


if __name__ == "__main__":
    main()
