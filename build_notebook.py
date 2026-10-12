"""Rebuild the standalone Kaggle notebook from tested local Python modules.

Usage: python build_notebook.py
No project-local imports or source-file dependencies remain in the notebook.
"""
import ast
import hashlib
import json
from pathlib import Path
import textwrap
import nbformat

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "detailtrust_kaggle.ipynb"
MODULES = ("config", "data", "model", "losses", "metrics", "evaluation", "training", "experiments", "reporting", "visuals", "checks")


def build_notebook(destination=OUTPUT):
    cells, audit, imports = [], [], []
    sources = {name: (ROOT/"detailtrust"/(name+".py")).read_text(encoding="utf-8") for name in MODULES}
    for source in sources.values():
        for node in ast.parse(source).body:
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom) and not node.level:
                line = ast.get_source_segment(source, node)
                if line not in imports:
                    imports.append(line)

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(text, name):
        cell = nbformat.v4.new_code_cell(textwrap.dedent(text).strip())
        cell["id"] = hashlib.sha256(name.encode()).hexdigest()[:12]
        cell["metadata"]["detailtrust_cell"] = name
        cells.append(cell)
        audit.append(dict(name=name, cell=len(cells), cell_id=cell["id"]))

    def module(name, heading):
        md(heading)
        source = sources[name]
        lines = source.splitlines()
        for node in ast.parse(source).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)) or isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
            segment = lines[start:node.end_lineno]
            # Local imports can also appear inside test functions. Their globals are
            # already defined in previous cells; preserve all other source lines.
            segment = [line for line in segment if not line.lstrip().startswith("from .")]
            node_name = getattr(node, "name", f"constants_{node.lineno}")
            doc = ast.get_docstring(node) if isinstance(node, (ast.ClassDef, ast.FunctionDef)) else None
            if doc:
                md(f"**`{node_name}`** — {doc}")
            code("\n".join(segment), name+"."+node_name)

    md("""
    # DetailTrust-Net: an executable BraTS research notebook

    This notebook implements the full architecture specified in `arch.md`: a
    five-level 3D residual U-Net, a learned three-scale Gaussian split, shared
    zero-anchored corrections and local utility-supervised gates. It contains all
    source code in executable cells. **No accompanying `.py` files are needed.**

    **Question:** do utility targets improve detail refinement beyond the same
    spatial gates trained with segmentation loss alone? B5 versus B4 is the key
    controlled comparison. This is a hypothesis, not an established improvement.

    Upload this notebook, attach the [BraTS 2020 NIfTI mirror](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation),
    enable a GPU, and choose **Save Version → Run All**. Default mode is a pipeline
    smoke run on real data; select `pilot` or `main` after it passes. Inputs must be
    the NIfTI mirror, not the similarly named HDF5 slice dataset. The notebook reads
    `/kaggle/input` and never requests or reads an API token.

    **Evidence boundary:** local CPU tests are reported separately in the delivery
    audit. No Kaggle GPU training result is embedded in this unexecuted notebook.
    Version, licence metadata and the actual mounted inventory are not pre-certified.
    Check the dataset card and [original BraTS terms](https://www.med.upenn.edu/cbica/brats2020/data.html).
    Official challenge validation/test labels are withheld; evaluation here uses a
    saved custom patient split of the labelled training collection.
    """)
    md("## A — Dependencies and reproducibility\nInstall only missing libraries. Preserve Kaggle's existing CUDA-enabled PyTorch. No credentials enter this cell.")
    code('''
        import importlib.util
        import subprocess
        import sys
        missing = [name for name in ("numpy", "scipy", "nibabel", "matplotlib", "nbformat")
                   if importlib.util.find_spec(name) is None]
        if missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("Select Kaggle's GPU Python image with PyTorch installed.")
        ''', "dependencies")
    code("\n".join(imports), "central_imports")
    module("config", "### Named settings and experiment registry\nEvery mode keeps widths 32/64/128/256/320. Updates are sampled patches; an epoch is a logging interval.")
    code('''
        # The main settings you edit before Save Version / Run All.
        MODE = "smoke"                # smoke, pilot, main, extended
        EXPERIMENTS = ("B0", "B1", "B2", "B4", "B5")
        SYNTHETIC_TEST_ONLY = False    # True ONLY for pipeline debugging, never research
        RUN_FINAL_TEST = False        # keep False during development
        PROTOCOL_LOCKED = False       # must be True before final test evaluation
        OUTPUT_DIRECTORY = "/kaggle/working/detailtrust"
        RESUME_DIRECTORY = None       # attached prior output, e.g. /kaggle/input/my-run/detailtrust
        cfg = configuration(MODE, variants=EXPERIMENTS, output_dir=OUTPUT_DIRECTORY,
                            synthetic=SYNTHETIC_TEST_ONLY, resume_dir=RESUME_DIRECTORY,
                            run_final_test=RUN_FINAL_TEST, protocol_locked=PROTOCOL_LOCKED)
        if SYNTHETIC_TEST_ONLY:
            cfg = replace(cfg, patch=(16, 16, 16), precision="float32", profile_updates=1,
                          profile_warmup=0, max_hours=1, val_subjects=1)
        seed_everything(cfg.seed)
        print({k: getattr(cfg, k) for k in ("mode", "patch", "train_subjects", "variants", "precision")})
        ''', "configuration")
    code('''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda" and not SYNTHETIC_TEST_ONLY:
            raise RuntimeError("Enable a Kaggle GPU for real-data training. CPU is supported only for explicit synthetic tests.")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        print("Training device:", device)
        if device.type == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))
            print("Memory GiB:", torch.cuda.get_device_properties(0).total_memory / 2**30)
        ''', "gpu_detection")
    code('''
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "figures").mkdir(exist_ok=True)
        if cfg.resume_dir:
            previous = Path(cfg.resume_dir)
            assert previous.is_dir() and previous.resolve() != output_dir.resolve()
            for name in ("experiments",):
                if (previous/name).exists():
                    shutil.copytree(previous/name, output_dir/name, dirs_exist_ok=True)
            for name in ("split_manifest.json",):
                if (previous/name).exists():
                    shutil.copy2(previous/name, output_dir/name)
        write_json(output_dir / "experiment_config.json", asdict(cfg))
        ''', "output_and_resume")
    code('''
        runtime_environment = environment_record(output_dir)
        write_json(output_dir / "environment.json", runtime_environment)
        print(runtime_environment)
        ''', "environment_storage")
    code('''
        # A failed early data check still leaves an honest status artifact.
        write_json(output_dir / "detailtrust_report.json", {
            "schema_version": "1.0.0", "experiment_status": "initializing",
            "training_configuration": asdict(cfg), "environment": runtime_environment,
            "validation_metrics": None, "test_metrics": None,
            "limitations": ["No experiment completed yet; final report replaces this initialization record."]})
        ''', "initialize_metadata")

    module("data", """## B — Dataset and preprocessing
    Match subjects before splitting. Original labels 0/1/2/4 map to internal 0/1/2/3.
    NiBabel reorients arrays to RAS without interpolating labels; the adapter checks
    1 mm spacing and affine alignment. A different spacing raises an error rather
    than silently changing the physical Gaussian scale. Crop boundaries are based
    only on available MRI. Tumour labels are used only for training-patch sampling
    and supervision. Each patch includes a four-voxel filtering halo.
    """)
    module("visuals", "### Small visualization helpers\nThese figures use a prespecified middle slice; they do not select the most flattering prediction.")
    code('''
        DATASET_ROOT = None           # None discovers compatible mounted inputs
        DATASET_VERSION = None        # fill with the version shown in Kaggle Input
        DATASET_LICENSE_NOTE = "Mirror licence not independently verified; consult dataset card and original BraTS research/citation terms."
        if not cfg.synthetic:
            cfg = replace(cfg, data_root=str(discover_dataset_root(DATASET_ROOT)), dataset_version=DATASET_VERSION)
            print("Verified existing mount:", cfg.data_root)
        ''', "dataset_path")
    code('''
        if cfg.synthetic:
            store = SyntheticStore()
            subjects = [dict(id=s, patient_id=s) for s in store.records]
            discovery = dict(eligible=len(subjects), synthetic=True, excluded=[])
        else:
            subjects, discovery = discover_subjects(cfg.data_root)
        write_json(output_dir / "dataset_discovery.json", discovery)
        print("Usable labelled subjects:", len(subjects), "Excluded:", len(discovery["excluded"]))
        ''', "subject_inventory")
    code('''
        if cfg.synthetic:
            integrity = dict(status="synthetic_only", subjects=len(subjects))
        else:
            # Reads each labelled case once; can take several minutes. Progress is
            # saved after every subject, including failures.
            integrity = inventory_geometry(subjects, output_dir / "dataset_integrity.json")
            store = SubjectStore(subjects, output_dir / "cache", cfg.cache_gb)
        write_json(output_dir / "dataset_integrity.json", integrity)
        ''', "geometry_label_audit")
    code('''
        split = make_split(subjects, cfg.split_seed)
        existing_manifest = output_dir / "split_manifest.json"
        if existing_manifest.exists():
            assert json.loads(existing_manifest.read_text())["sha256"] == split["sha256"], "Dataset/split changed during resume"
        assert_split(split)
        write_json(existing_manifest, split)
        print({k: len(split[k]) for k in ("train", "validation", "test")}, "CUSTOM split")
        ''', "subject_split")
    code('''
        train_ids = split["train"][:min(cfg.train_subjects, len(split["train"]))]
        val_ids = split["validation"][:cfg.val_subjects]
        example_id = train_ids[0]
        example_images, example_labels, example_geometry = store.load(example_id)
        print("Example:", example_id, example_images.shape, np.unique(example_labels), example_geometry["spacing"])
        plot_subject(example_images, example_labels, output_dir / "figures" / "preprocessing.png", "Normalized training example; labels never enter input normalization")
        ''', "preprocessing_example")
    code('''
        dataset_evidence = dict(identifier=DATASET_SLUG if not cfg.synthetic else "synthetic_execution_fixture",
            source="https://www.kaggle.com/datasets/"+DATASET_SLUG, version=cfg.dataset_version,
            version_status="user_declared" if cfg.dataset_version else "not independently verified",
            licence=DATASET_LICENSE_NOTE, original_terms="https://www.med.upenn.edu/cbica/brats2020/data.html",
            discovery=discovery, official_test_labels_used=False, mounted_files_verified=not cfg.synthetic,
            patient_identity_limit="one BraTS20 subject ID per patient; no cross-release pooling",
            inventory_hash=file_hash(output_dir / "dataset_integrity.json"))
        write_json(output_dir / "dataset_evidence.json", dataset_evidence)
        write_json(output_dir / "experiment_config.json", asdict(cfg))
        ''', "dataset_evidence")
    code('''
        patch_dataset = PatchDataset(store, train_ids, cfg.patch, cfg.seed, 2)
        loader = make_loader(patch_dataset)
        example_batch = next(iter(loader))
        assert example_batch["images"].shape == (1, 4, *(n+8 for n in cfg.patch))
        assert example_batch["labels"].shape == (1, *cfg.patch)
        assert set(example_batch["labels"].unique().tolist()) <= {0, 1, 2, 3}
        print({k: tuple(v.shape) for k, v in example_batch.items() if isinstance(v, torch.Tensor)})
        ''', "dataloader_shape_check")

    module("model", """## C/D — Baseline and DetailTrust-Net
    The raw baseline uses four MRI channels plus four availability channels. The
    same five-level residual U-Net also supplies the proposed model's base logits
    and 32-channel features. The model has **16,934,644 parameters** in B5.

    `L_m = Σ softmax(α_m)_s G_σs(X_m)` with σ = 0.5/0.8/1.2 **mm**; `H=X-L`.
    `r_m=R(F,H_m,e_m)-R(F,0,e_m)` uses 32+1+4=37 input channels.
    The gate receives 32+1+3+4+4=44 channels. Final foreground logits are
    `z=z0+Σ a*q*r`, and the base background logit is unchanged. There is no later
    convolution. A gate is a learned multiplier, not a calibrated quality probability.
    """)
    code('''
        reference_model = DetailTrustNet(checkpoint_blocks=cfg.checkpoint_blocks).to(device)
        print(parameter_counts(reference_model))
        assert parameter_counts(reference_model)["total"] == 16_934_644
        with torch.no_grad():
            small_x = torch.randn(1, 4, 24, 24, 24, device=device)
            small_a = torch.tensor([[1., 1., 1., 1.]], device=device)
            small_result = reference_model(small_x, small_a)
        print({k: tuple(v.shape) for k, v in small_result.items()})
        torch.testing.assert_close(small_result["logits"][:, 0], small_result["base"][:, 0])
        del reference_model, small_result, small_x
        if device.type == "cuda": torch.cuda.empty_cache()
        ''', "reference_model_shapes")
    module("losses", """## E — Training objectives
    A: train the base and smoothing with CE plus WT/TC/ET soft-Dice.
    B: gates equal one; independently drop detail branches with probability .25;
    use final segmentation + .3 base segmentation + .1 available-branch auxiliary mean.
    C: freeze the candidates and train gates. In the all-candidates-present context,
    `delta=CE(z_all-r_m,Y)-CE(z_all,Y)`. Positive means helpful. Average over 3³,
    divide by .1, apply sigmoid, and **detach**. Gate BCE gives equal weight to WT
    and background when both exist. No training label enters an inference feature.
    """)
    module("metrics", """## F — Metrics and conditional stability
    Report patient macro averages, preserve empty cases, and use physical millimetres
    for surfaces. The stability bound considers attenuation of already-computed
    contributions only. It does not certify correctness, input corruption, or
    whole-modality removal. One-empty HD95 is undefined (`null`), not zero.
    """)
    module("evaluation", """### Full-volume evaluation and explanation accounting
    Blend logits, base scores and contributions with identical window weights.
    Apply softmax/argmax only after blending. All 15 nonempty MRI subsets are
    distinct from the 16 detail-retention subsets used to check the bound.
    Degraded validation uses development corruptions; final testing adds held-out
    ghost/ringing families. All variants get identical deterministic conditions.
    """)
    module("training", """## G — GPU training, checkpointing and resumption
    CUDA uses float16 autocast and gradient scaling; losses reduce in float32.
    Checkpoint large blocks and frozen Stage C candidates to manage memory.
    No width reduction or automatic patch-size change is permitted. Latest checkpoints
    preserve optimizer/scheduler/scaler/RNG state; best checkpoints use validation
    macro Dice. Pause at the time cap and resume from attached outputs.
    """)
    module("experiments", """## H — Experiment execution
    The registry controls every variant. B0 and B1 train for the full update budget.
    B2/B3/B4/B5/A1/A6 reuse the same A/B candidate checkpoint, with matched gate
    initialization for B4/B5/A1. B2 gets an extra B-style refinement interval so
    the total optimizer-update count matches. This does not imply equal GPU cost.
    Secondary ablations are fully executable; unselected runs receive `not_run`.
    """)
    module("reporting", """## I — Report and artifact generation
    JSON uses null for undefined values and includes the reason/count. CSV and
    figures contain actual records only. A missing trained checkpoint is recorded
    as missing, never replaced with random weights named 'best'.
    """)
    module("checks", """## J — Tests before expensive training
    Synthetic tensors test geometry-independent model algebra and gradients.
    The NIfTI test writes a tiny reoriented volume and exercises the actual adapter.
    These are correctness checks, not evidence of tumour-segmentation accuracy.
    """)
    code('''
        checks = run_unit_checks()
        write_json(output_dir / "unit_checks.json", checks)
        assert all(item["status"] == "passed" for item in checks), "Resolve blocked tests before Kaggle training."
        ''', "execute_unit_checks")
    code('''
        overfit_check = tiny_overfit_check(store, train_ids, cfg, device, updates=5 if cfg.mode == "smoke" else 20)
        write_json(output_dir / "tiny_overfit_check.json", overfit_check)
        print("Repeated patch loss decreased:", overfit_check["loss_decreased"])
        # A short run need not fully memorize the patch. Inspect this curve if training fails.
        ''', "tiny_overfit_execution")
    code('''
        profiles = profile_training_stages(store, train_ids, cfg, experiment_registry()["B5"], device)
        write_json(output_dir / "compute_profile.json", profiles)
        print(profiles)
        estimated_hours = sum(n * p["median_step_seconds"] for n, p in zip(
            (cfg.updates_a, cfg.updates_b, cfg.updates_c), profiles)) / 3600
        print("Measured-step extrapolation for ONE B5 run, excluding validation:", round(estimated_hours, 2), "hours")
        ''', "gpu_profile_execution")
    md("### Run the selected experiments\nThis cell can run for hours. It checkpoints frequently and records paused/failed variants. Resume in a later session; do not treat a partial sweep as completed evidence.")
    code('''
        statuses = run_experiment_sweep(store, split, cfg, device)
        print({name: item["status"] for name, item in statuses.items()})
        ''', "run_experiments")
    code('''
        best_checkpoint = export_best_detailtrust(statuses, output_dir, device)
        print(best_checkpoint)
        ''', "best_checkpoint_export")
    code('''
        if best_checkpoint["status"] == "saved":
            state = torch.load(best_checkpoint["path"], map_location=device, weights_only=False)
            best_model = DetailTrustNet(**state["architecture"]).to(device)
            best_model.load_state_dict(state["model_state"], strict=True)
            chosen_subject = val_ids[0]  # prespecified, never chosen for a flattering result
            shown_images, shown_labels, _ = store.load(chosen_subject)
            shown_output = sliding_window(best_model, shown_images, np.ones(4, np.float32), cfg, device, analysis=True)
            plot_corrections(shown_images, shown_labels, shown_output, output_dir / "figures" / "example_corrections.png")
            del best_model, shown_output, state
            if device.type == "cuda": torch.cuda.empty_cache()
        ''', "example_contribution_figure")
    code('''
        report, records, summary = build_report(cfg, split, integrity, statuses, best_checkpoint, checks, profiles, dataset_evidence)
        artifacts = export_report(report, records, summary, output_dir)
        print("Report:", output_dir / "detailtrust_report.json")
        ''', "report_export")
    readme = (ROOT/"README-detailtrust.md").read_text(encoding="utf-8") if (ROOT/"README-detailtrust.md").exists() else "Run the notebook top to bottom. All actual results are in detailtrust_report.json."
    code(f'(output_dir / "README.md").write_text({readme!r}, encoding="utf-8")', "save_runtime_readme")
    code('''
        artifact_verification = verify_artifacts(output_dir)
        for name, result in artifact_verification.items():
            print(name, "saved" if result["exists"] else "NOT PRODUCED — inspect experiment status")
        ''', "verify_artifacts")
    code('''
        with open(output_dir / "detailtrust_report.json", encoding="utf-8") as handle:
            saved_report = json.load(handle)
        assert set(REPORT_FIELDS) <= set(saved_report)
        print("Evidence level:", saved_report["project"]["evidence_level"])
        print("Best checkpoint:", saved_report["best_checkpoint"])
        print("Final test:", saved_report["test_metrics"]["status"])
        print("Experiments:", saved_report["experiment_status"])
        ''', "read_report_example")
    md("""## How to interpret the run
    A smoke test establishes execution only. A pilot is preliminary and may be
    undertrained. Inspect B5 versus B4, clean ET, contribution activity, repaired
    and introduced errors, and stable wrong predictions. Do not use a high stability
    fraction as evidence that the segmentation is correct.

    Keep test evaluation disabled until all methods, budgets and conditions are
    frozen. Main experiments may span multiple Kaggle quotas. Three seeds and
    external validation remain necessary for stronger claims. The notebook does
    not reproduce the published M²FTrans, evidential fusion or SwitchNet systems;
    its internal ablations alone cannot support state-of-the-art claims.

    Sources: [BraTS protocol and citations](https://www.med.upenn.edu/cbica/brats2020/data.html),
    [NiBabel orientation](https://nipy.org/nibabel/image_orientation.html),
    [PyTorch checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html),
    [Kaggle GPU guidance](https://www.kaggle.com/docs/efficient-gpu-usage).
    """)
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "detailtrust": {"arch_sha256": hashlib.sha256((ROOT/"arch.md").read_bytes()).hexdigest(),
                        "self_contained": True, "source_hashes": {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()}}})
    nbformat.validate(notebook)
    for cell in cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert not any(isinstance(n, ast.ImportFrom) and n.level for n in ast.walk(ast.parse(cell.source)))
    nbformat.write(notebook, destination)
    (ROOT/"notebook_cell_map.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Built {destination.name}: {len(cells)} cells, {sum(c.cell_type == 'code' for c in cells)} code cells")
    return destination


if __name__ == "__main__":
    build_notebook()
