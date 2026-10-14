import os
import glob
import json

def merge_results(parts_dir="experiments/out/gnn_parts", output_file="experiments/out/gnn_results.json"):
    merged = {}
    json_paths = sorted(glob.glob(os.path.join(parts_dir, "*", "gnn_results.json")))
    
    if not json_paths:
        print(f"No result files found in {parts_dir}")
        return

    for path in json_paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)
                for run_key, topologies in data.items():
                    if run_key not in merged:
                        merged[run_key] = {}
                    merged[run_key].update(topologies)
            print(f"Loaded: {path}")
        except Exception as e:
            print(f"Error reading {path}: {e}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(merged, f, indent=4)
        
    print(f"\nSuccessfully merged {len(json_paths)} runs into {output_file}")
    print("Contained configurations:", list(merged.keys()))

if __name__ == "__main__":
    merge_results()
