#!/bin/bash

# Generate questions in batches

# OP_DIR=/generated_questions
# start_=1
# stop_=25
# mult=5000
# for ((i=$start_;i<=$stop_;i++));do
# 	ch_start=$((i*mult-mult))
# 	ch_end=$((i*mult))
#         tests="test"$((i-1))"s"
# 	screen -dmS $tests sh -c "python generate_questions.py ../premises/vqa_oe_tuples_filtered.json $OP_DIR/op_$ch_start_$ch_end.json $ch_start $ch_end;exec /bin/bash"
# done

OP_DIR=./generated_questions
start_=1
stop_=2
mult=5
for ((i=$start_;i<=$stop_;i++));do
	ch_start=$((i*mult-mult))
	ch_end=$((i*mult))
        tests="test"$((i-1))"s"
	screen -d -m -S $tests sh -c "python generate_questions.py ../premises/vqa_oe_tuples_filtered.json $OP_DIR/op_$ch_start_$ch_end.json $ch_start $ch_end;"
done