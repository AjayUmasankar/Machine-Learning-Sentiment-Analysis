#!/usr/bin/env bash

TEST_DIR="./tests"

total_lines=0
total_lines_different=0
for tk in "$TEST_DIR/"*; do
    echo "running test '$tk'..."
    for pgrm in "$tk/"*; do
        if [ -d "$pgrm" ]; then
            pgrm_file="`basename $pgrm`.py"
            python3 "$pgrm_file" "$tk/trainset.tsv" "$tk/testset.tsv" > "$pgrm/your_output.txt"
            n_lines="`cat "$pgrm/sample_output.txt" | wc -l`"
            n_lines_different="$(diff -y --suppress-common-lines "$pgrm/sample_output.txt" "$pgrm/your_output.txt" | wc -l)"
            echo "Your program $pgrm_file produced $n_lines_different/$n_lines lines different from my output."
            total_lines=`expr $total_lines + $n_lines`
            total_lines_different=`expr $total_lines_different + $n_lines_different`
        fi
    done
    echo ""
done

printf "Summary: "
if [ "$total_lines_different" -eq 0 ]; then
    echo "Your outputs are identical with my outputs ($total_lines lines)"
else
    echo "$total_lines_different/$total_lines lines are different"
fi
