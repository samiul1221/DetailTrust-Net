"""Execute generated cells with explicit CPU synthetic substitutions, never BraTS claims."""
from pathlib import Path
import contextlib
import io
import json
import time
import traceback
import tempfile
import nbformat
from build_notebook import build_notebook


def main():
    path = build_notebook()
    notebook = nbformat.read(path, as_version=4)
    temporary = tempfile.TemporaryDirectory(prefix='detailtrust-notebook-test-')
    namespace = {"__name__": "__main__", "VERIFICATION_OUTPUT": temporary.name}
    records = []
    substitutions = {
        "dependencies": "# Installation skipped: network blocked; existing CPU dependencies used.",
        "configuration": '''
MODE = "smoke"
SYNTHETIC_TEST_ONLY = True
RESUME_BUNDLE_ID = None
cfg = configuration(MODE, output_dir=VERIFICATION_OUTPUT,
    patch=(16,16,16), precision="float32", synthetic=True, updates_a=1,
    updates_b=1, updates_c=1, validation_every=1, epoch_updates=1,
    profile_updates=1, profile_warmup=0, max_hours=1.)
seed_everything(cfg.seed)
''',
        "execute_unit_checks": '''
checks = run_unit_checks()
write_json(output_dir / "unit_checks.json", checks)
assert all(c["status"] == "passed" or c["name"] == "check_nifti_adapter" and c["status"] == "blocked" for c in checks)
''',
    }
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        name = cell.metadata["detailtrust_cell"]
        source = substitutions.get(name, cell.source)
        if name == "subject_inventory":
            source = source.replace("SyntheticStore()", "SyntheticStore(size=16)")
        record = dict(cell=name, modified_for_cpu=source != cell.source)
        started = time.perf_counter()
        stream = io.StringIO()
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                exec(compile(source, f"notebook::{name}", "exec"), namespace)
            record["status"] = "passed"
        except Exception:
            record["status"] = "failed"
            record["error"] = traceback.format_exc()
        record["seconds"] = time.perf_counter()-started
        record["output"] = stream.getvalue()
        records.append(record)
        print(name, record["status"], flush=True)
        Path("verification/notebook_execution.json").write_text(json.dumps({
            "evidence": "CPU synthetic cell execution, not real-data/CUDA verification",
            "substitutions": list(substitutions)+["subject_inventory"], "cells": records}, indent=2), encoding="utf-8")
        if record["status"] == "failed":
            raise RuntimeError(record["error"])
    statuses = namespace["statuses"]
    assert all(statuses[f"{v}_s1729"]["status"] == "completed" for v in namespace["cfg"].variants)
    print("All notebook cells executed with documented CPU substitutions.")
    temporary.cleanup()


if __name__ == "__main__":
    main()
