import json
import math
import sys


def median_cpu_times(path):
    with open(path, encoding="utf-8") as result_file:
        results = json.load(result_file)

    benchmarks = results.get("benchmarks", [])
    medians = {
        entry["name"]: entry["cpu_time"]
        for entry in benchmarks
        if entry.get("aggregate_name") == "median"
    }
    if not medians:
        raise ValueError(f"No median benchmark results found in {path}")
    return medians


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: compare.py BASELINE.json CURRENT.json")

    baseline = median_cpu_times(sys.argv[1])
    current = median_cpu_times(sys.argv[2])
    matched_names = sorted(baseline.keys() & current.keys())
    if not matched_names:
        raise ValueError("No matching benchmark names between baseline and current results")

    print("## Benchmark comparison")
    print()
    print("Lower CPU time is faster. Values and changes use each benchmark's median.")
    print()
    print("| Benchmark | Baseline (ns) | Current (ns) | Change |")
    print("|---|---:|---:|---:|")
    for name in matched_names:
        baseline_time = baseline[name]
        current_time = current[name]
        if not math.isfinite(baseline_time) or baseline_time <= 0:
            raise ValueError(f"Invalid baseline CPU time for {name}: {baseline_time}")
        if not math.isfinite(current_time) or current_time <= 0:
            raise ValueError(f"Invalid current CPU time for {name}: {current_time}")
        change = (current_time / baseline_time - 1) * 100
        safe_name = name.replace("|", "\\|")
        print(
            f"| {safe_name} | {baseline_time:.2f} | "
            f"{current_time:.2f} | {change:+.1f}% |"
        )

    unmatched = (baseline.keys() | current.keys()) - (baseline.keys() & current.keys())
    if unmatched:
        print()
        print("Benchmarks present in only one run: " + ", ".join(sorted(unmatched)))


if __name__ == "__main__":
    main()
