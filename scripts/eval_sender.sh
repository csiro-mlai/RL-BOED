folder=$1
for val in $(seq 1000 500 40000)
do
    src="$folder"itr_"$val".pkl
    dest="$folder"evaluation_"$val".log
    sbatch eval_run.sh $folder $val
done
