import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

TESTS = [
    "accuracy.py",
    "latency.py",
    "accuracy_tiled_batched.py",
    "latency_tiled_batched.py",
]


def main():

    print("=" * 70)
    print("D-FINE FULL BENCHMARK")
    print("=" * 70)

    failures = []

    for index, script_name in enumerate(
        TESTS,
        start=1,
    ):

        script_path = ROOT / script_name

        if not script_path.exists():

            print()
            print(f"MISSING: {script_name}")

            failures.append(
                script_name
            )

            continue

        print()
        print("=" * 70)
        print(
            f"TEST {index}/{len(TESTS)}: "
            f"{script_name}"
        )
        print("=" * 70)

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
            ],
            cwd=ROOT,
        )

        if result.returncode != 0:

            print()
            print(
                f"FAILED: {script_name} "
                f"(exit code "
                f"{result.returncode})"
            )

            failures.append(
                script_name
            )

        else:

            print()
            print(
                f"COMPLETE: "
                f"{script_name}"
            )

    print()
    print("=" * 70)
    print("ALL TESTS FINISHED")
    print("=" * 70)

    if failures:

        print("Failures:")

        for script_name in failures:
            print(f"  - {script_name}")

        sys.exit(1)

    print(
        "All 4 tests completed "
        "successfully."
    )


if __name__ == "__main__":
    main()
