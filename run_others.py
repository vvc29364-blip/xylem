"""Re-run every job that failed in the last `run_all.py` run (and then rebuild the report).

    python run_others.py                 # all failed jobs on all visible GPUs
    python run_others.py --gpus 0,1      # any extra args are passed through to run_all.py

Finished jobs (results/_jobs/*.done) are skipped; only the *.failed ones (plus the final report,
so the tables include the new numbers) are executed again.
"""
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    budget = "full"
    if "--budget" in sys.argv:
        budget = sys.argv[sys.argv.index("--budget") + 1]
    jd = os.path.join(ROOT, "results" if budget == "full" else f"results_{budget}", "_jobs")
    failed = sorted(glob.glob(os.path.join(jd, "*.failed")))
    print(f"{len(failed)} failed job marker(s) in {jd}", flush=True)
    for f in failed:
        os.remove(f)
    report = os.path.join(jd, "report.done")
    if failed and os.path.exists(report):
        os.remove(report)          # regenerate tables/figures with the new results
    cmd = [sys.executable, os.path.join(ROOT, "run_all.py"), "--retry_failed"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
