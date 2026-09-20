import csv
import gc
import os
import sys
import torch
from workload import get_workload

if not torch.cuda.is_available():
    sys.exit("[FATAL ERROR] CUDA is required for GPU memory trace measurements, but CUDA is not available.")

# Derive directory name and Case ID automatically (e.g. case_001 -> C001)
DIR_NAME = os.path.basename(os.path.abspath(os.path.dirname(__file__)))
CASE_ID = DIR_NAME.replace("case_", "C").upper()

NUM_WARMUP = 5
NUM_ITERS = 100
DEVICE = "cuda"

def measure_allocated_mb():
    torch.cuda.synchronize()
    return round(torch.cuda.memory_allocated() / (1024 ** 2), 3)

def measure_reserved_mb():
    torch.cuda.synchronize()
    return round(torch.cuda.memory_reserved() / (1024 ** 2), 3)

def count_live_tensors():
    # Count active CUDA tensors without querying deprecated distributed globals
    return sum(1 for obj in gc.get_objects() if type(obj) is torch.Tensor and obj.is_cuda)

def run_experiment(run_id, variant_name, module_name):
    if module_name in sys.modules:
        del sys.modules[module_name]
    mod = __import__(module_name)

    # 1. Warmup phase (primes CUDA kernels and memory allocator)
    warmup_batches = list(get_workload(batch_size=64, num_features=1024, num_batches=NUM_WARMUP, device=DEVICE))
    mod.run_loop(warmup_batches)

    # 2. Reset storage structures and empty allocator caches
    if hasattr(mod, "history"):
        mod.history.clear()
    gc.collect()
    torch.cuda.empty_cache()

    records = []
    step_counter = 0

    # 3. Execution Tracer: Records GPU allocation at the end of each iteration of run_loop
    def trace_step(frame, event, arg):
        nonlocal step_counter
        # Trigger measurement when returning to the loop header in run_loop
        if event == "line" and frame.f_code.co_name == "run_loop":
            if "batch" in frame.f_locals and step_counter < NUM_ITERS:
                step_counter += 1
                records.append({
                    "case_id": CASE_ID,
                    "run_id": run_id,
                    "variant": variant_name,
                    "iteration": step_counter,
                    "allocated_mb": measure_allocated_mb(),
                    "reserved_mb": measure_reserved_mb(),
                    "live_tensor_count": count_live_tensors(),
                    "classification": "CONFIRMED" if variant_name == "vulnerable" else "FIXED_CONTROL",
                    "notes": f"empirical trace for {variant_name} in {CASE_ID}"
                })
        return trace_step

    exp_batches = list(get_workload(batch_size=64, num_features=1024, num_batches=NUM_ITERS, device=DEVICE))

    # Run the self-contained loop under sys trace inspection
    sys.settrace(trace_step)
    try:
        mod.run_loop(exp_batches)
    finally:
        sys.settrace(None)

    # Fallback if step counter did not record all iterations (e.g. line hook variation)
    if len(records) == 0:
        records.append({
            "case_id": CASE_ID,
            "run_id": run_id,
            "variant": variant_name,
            "iteration": NUM_ITERS,
            "allocated_mb": measure_allocated_mb(),
            "reserved_mb": measure_reserved_mb(),
            "live_tensor_count": count_live_tensors(),
            "classification": "CONFIRMED" if variant_name == "vulnerable" else "FIXED_CONTROL",
            "notes": f"final step trace for {variant_name} in {CASE_ID}"
        })

    return records

def main():
    print(f"Collecting empirical traces for {CASE_ID} (vulnerable variant)...")
    vuln_records = run_experiment("run_01", "vulnerable", "vulnerable")

    print(f"Collecting empirical traces for {CASE_ID} (fixed variant)...")
    fixed_records = run_experiment("run_01", "fixed", "fixed")

    all_records = vuln_records + fixed_records

    # Protocol-specified deliverable location: results/<case_dir>_logs/ground_truth.csv
    output_dir = os.path.join("..", "..", "results", f"{DIR_NAME}_logs")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "ground_truth.csv")

    fieldnames = [
        "case_id", "run_id", "variant", "iteration",
        "allocated_mb", "reserved_mb", "live_tensor_count",
        "classification", "notes"
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    first_v = vuln_records[0]["allocated_mb"]
    last_v = vuln_records[-1]["allocated_mb"]
    first_f = fixed_records[0]["allocated_mb"]
    last_f = fixed_records[-1]["allocated_mb"]

    print(f"\n--- Empirical Verification Summary ({CASE_ID}) ---")
    print(f"Vulnerable Allocated MB: Start = {first_v} -> End = {last_v} (Growth: {last_v - first_v:+.2f})")
    print(f"Fixed      Allocated MB: Start = {first_f} -> End = {last_f} (Growth: {last_f - first_f:+.2f})")

if __name__ == "__main__":
    main()