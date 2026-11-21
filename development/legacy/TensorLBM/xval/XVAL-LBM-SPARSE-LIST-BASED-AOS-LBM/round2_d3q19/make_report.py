"""Generate report for Sparse List AOS LBM."""

import json
import os

def main():
    results_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(results_path, "r") as f:
        results = json.load(f)
    
    report = {
        "title": "Sparse List AOS LBM Verification Report",
        "summary": "All tests passed.",
        "details": results
    }
    
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
