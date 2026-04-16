"""Run the test scenario and print step-by-step node outputs."""
import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from datetime import datetime, timezone
from src.agents.graph import create_default_graph

SCENARIO = {
    "voltage_pu": 0.78,
    "current_pu": 1.45,
    "frequency_hz": 49.5,
    "bus_id": "BUS_001",
    "feeder_id": "F1",
    "timestamp": datetime.now(timezone.utc),
}

def main():
    graph = create_default_graph()

    print(f"\n{'='*80}")
    print(f"  TEST SCENARIO: voltage_pu=0.78, current_pu=1.45, frequency_hz=49.5")
    print(f"  Bus: BUS_001, Feeder: F1")
    print(f"{'='*80}\n")

    # Use .stream() to see each node's output step by step
    for i, step in enumerate(graph.stream(SCENARIO), 1):
        for node_name, node_output in step.items():
            print(f"--- Step {i}: Node '{node_name}' ---")
            # Pretty-print the output, handling non-serializable types
            for key, value in sorted(node_output.items()):
                if key == "agent_trace":
                    print(f"  {key}:")
                    for entry in value:
                        print(f"    - node: {entry.get('node')}")
                        print(f"      action: {entry.get('action')}")
                        if 'outputs' in entry:
                            for k, v in entry['outputs'].items():
                                print(f"      {k}: {v}")
                elif isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        if isinstance(v, list) and len(str(v)) > 60:
                            print(f"    {k}: [{len(v)} items]")
                        else:
                            print(f"    {k}: {v}")
                elif isinstance(value, list):
                    print(f"  {key}: {value}")
                elif isinstance(value, datetime):
                    print(f"  {key}: {value.isoformat()}")
                else:
                    print(f"  {key}: {value}")
            print()

    # Final state
    result = graph.invoke(SCENARIO)
    print(f"{'='*80}")
    print(f"  FINAL STATE SUMMARY")
    print(f"{'='*80}")
    print(f"  Fault Type:      {result.get('fault_type')}")
    print(f"  Severity:        {result.get('severity')}")
    print(f"  Confidence:      {result.get('confidence')}")
    print(f"  Fault ID:        {result.get('fault_id')}")
    print(f"  Affected:        {result.get('affected_components')}")
    print(f"  Iteration:       {result.get('iteration')}")
    print(f"  Trace entries:   {len(result.get('agent_trace', []))}")
    wf = result.get('wavelet_features', {})
    print(f"  Wavelet fault:   {wf.get('fault_detected')}")
    print(f"  Anomaly score:   {wf.get('anomaly_score')}")
    print(f"  Detail ratio:    {wf.get('detail_ratio')}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
