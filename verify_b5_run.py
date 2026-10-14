"""Exercise B5-only training and patient-resumable evaluation on tiny CPU fixtures."""
from pathlib import Path
import json
import tempfile
from unittest.mock import patch
import torch
from detailtrust.config import configuration, write_json
from detailtrust.data import SyntheticStore, make_split
from detailtrust.model import DetailTrustNet
from detailtrust.evaluation import evaluate_condition, evaluation_conditions, sliding_window
from detailtrust.visuals import plot_qualitative_paper
from detailtrust.experiments import run_experiment_sweep
from detailtrust.reporting import build_report, export_report
from detailtrust.checks import check_validation_selection_equivalence, check_patch_stream_and_metrics


def main():
    torch.set_num_threads(4)
    main_cfg = configuration("main")
    assert main_cfg.variants == ("B5",)
    assert (main_cfg.updates_a, main_cfg.updates_b, main_cfg.updates_c) == (4000, 4500, 1500)
    assert abs(main_cfg.max_hours * main_cfg.training_budget_fraction - 8.1) < 1e-9
    check_validation_selection_equivalence()
    check_patch_stream_and_metrics()
    store = SyntheticStore(count=20, size=16)
    split = make_split([dict(id=s) for s in store.records], main_cfg.split_seed)
    with tempfile.TemporaryDirectory(prefix="detailtrust-b5-test-") as temporary:
        cfg = configuration("smoke", output_dir=temporary, patch=(16, 16, 16),
            precision="float32", synthetic=True, updates_a=1, updates_b=1, updates_c=1,
            validation_every=1, epoch_updates=1, max_hours=1.)
        statuses = run_experiment_sweep(store, split, cfg, torch.device("cpu"))
        b5 = statuses["B5_s1729"]
        assert b5["status"] == "completed", b5.get("reason")
        assert [s["stage"] for s in b5["stages"]] == ["A", "B", "C"]
        assert all(v["status"] == "not_run" for k, v in statuses.items() if k != "B5_s1729")
        assert not (Path(temporary)/"experiments"/"B0_s1729").exists()
        checkpoint = Path(temporary)/"best_detailtrust_model.pt"
        assert checkpoint.exists()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = DetailTrustNet(**state["architecture"])
        model.load_state_dict(state["model_state"], strict=True)
        clean = evaluation_conditions(cfg)[0]
        assert clean["id"] == "clean_m15" and len(evaluation_conditions(cfg)) == 16
        subjects = split["validation"][:2]
        assert len(subjects) == 2
        clock = [0.]
        with patch("detailtrust.evaluation.time.perf_counter", side_effect=lambda: clock[0]):
            partial = evaluate_condition(model, store, subjects, split, "validation", clean,
                "resume_check", cfg, torch.device("cpu"), deadline=1.,
                save_progress=lambda result: clock.__setitem__(0, 2.))
        assert partial["status"] == "partial" and partial["completed_subjects"] == subjects[:1]
        # JSON round trip matches the persisted resume protocol.
        with patch.object(store, "load", wraps=store.load) as loading:
            resumed = evaluate_condition(model, store, subjects, split, "validation", clean,
                "resume_check", cfg, torch.device("cpu"), previous=json.loads(json.dumps(partial)))
            assert loading.call_count == 1 and loading.call_args.args[0] == subjects[1]
        complete = evaluate_condition(model, store, subjects, split, "validation", clean,
            "resume_check", cfg, torch.device("cpu"))
        assert resumed["status"] == "completed" and resumed["rows"] == complete["rows"]
        assert resumed["mean_loss"] == complete["mean_loss"]
        report, records, summary = build_report(cfg, split, {"status": "synthetic_only"},
            statuses, {"status": "saved", "path": str(checkpoint)}, [], [],
            {"identifier": "synthetic_engineering_fixture"})
        export_report(report, records, summary, temporary)
        assert json.loads((Path(temporary)/"detailtrust_report.json").read_text())["project"]["synthetic"]
        images, labels, metadata = store.load(subjects[0])
        output = sliding_window(model, images, __import__('numpy').ones(4, dtype='float32'), cfg, torch.device('cpu'), analysis=True)
        qualitative = plot_qualitative_paper(images, labels, output, Path(temporary)/'figures', subjects[0], metadata['spacing'])
        assert len(qualitative) == 3 and len(list((Path(temporary)/'figures').glob('*.pdf'))) >= 10
    write_json("verification/b5_only_checks.json", dict(status="passed",
        evidence="CPU synthetic engineering checks only; no medical-performance result",
        checks=["B5-only default", "unchanged 4000/4500/1500 main updates", "8.1 hour main training allowance",
                "same validation criterion", "A/B/C execution", "no baseline trained", "strict best checkpoint reload",
                "clean four-modality evaluation first", "patient resume without duplicates", "JSON/CSV/figure export"]))
    print("B5-only checks passed; temporary synthetic checkpoints removed.")


if __name__ == "__main__":
    main()
