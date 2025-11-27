#!/usr/bin/env python3
"""Extract evaluation results from training logs."""

from pathlib import Path

results_dir = Path(__file__).parent

for results_file in results_dir.glob("results*.txt"):
    output_file = results_dir / f"sweep_{results_file.name}"
    
    with open(results_file) as f:
        lines = f.readlines()
    
    # Extract sections from "Evaluation complete" to "wandb: "
    output_lines = []
    in_section = False
    
    for line in lines:
        if "## Evaluation complete" in line:
            in_section = True
        
        if in_section:
            output_lines.append(line)
        
        if in_section and line.strip() == "wandb:":
            in_section = False
    
    with open(output_file, 'w') as f:
        f.writelines(output_lines)
    
    print(f"Processed {results_file.name} -> {output_file.name}")
