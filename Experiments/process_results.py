import csv
import argparse
import numpy as np


def main(fpaths, dest):
	fpaths = fpaths.split(",")
	ameans = []
	astds = []
	rmedians = []
	rlqs = []
	ruqs = []
	rmeans = []
	pstds = []

	for fpath in fpaths:
		with open(fpath, newline='') as csvfile:
			reader = csv.DictReader(csvfile)
			amean = []
			astd = []
			rmedian = []
			rlq = []
			ruq = []
			rmean = []
			pstd = []
			for row in reader:
				if "Action/MeanAction" in row: amean.append(row["Action/MeanAction"])
				if "Action/StdAction" in row: astd.append(row["Action/StdAction"])
				if "Return/MedianReturn" in row: rmedian.append(row["Return/MedianReturn"])
				if "Return/LowerQuartileReturn" in row: rlq.append(row["Return/LowerQuartileReturn"])
				if "Return/UpperQuartileReturn" in row: ruq.append(row["Return/UpperQuartileReturn"])
				if "Evaluation/AverageReturn" in row: rmean.append(row["Evaluation/AverageReturn"])
				if "Policy/MeanStd" in row: pstd.append(row["Policy/MeanStd"])

			ameans.append(amean)
			astds.append(astd)
			rmedians.append(rmedian)
			rlqs.append(rlq)
			ruqs.append(ruq)
			rmeans.append(rmean)
			pstds.append(pstd)
	
	np.savez_compressed(
		dest, ameans=ameans, astds=astds, rmedians=rmedians, rlqs=rlqs,
		ruqs=ruqs, rmeans=rmeans, pstds=pstd
	)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="RL results parser")
	parser.add_argument("--fpaths", nargs="?", default="", type=str)
	parser.add_argument("--dest", default="results.npz", type=str)

	args = parser.parse_args()
	main(args.fpaths, args.dest)