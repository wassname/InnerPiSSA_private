
#!/bin/bash
for f in results*.txt; do
    # Extract base name without .txt extension
    base="${f%.txt}"
    # grep -B1 -A3 '🥇' "$f" --group-separator='' > "sweep_${base}.txt"
    # Get everything from "Evaluation complete" to the wandb summary line
    sed -n '/## Evaluation complete/,/^wandb: $/p' "$f" > "sweep_${base}.txt"
    echo "Processed $f -> sweep_${base}.txt"
done
