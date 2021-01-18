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

	for fpath in fpaths:
		with open(fpath, newline='') as csvfile:
			reader = csv.DictReader(csvfile)
			amean = []
			astd = []
			rmedian = []
			rlq = []
			ruq = []
			rmean = []
			for row in reader:
				amean.append(row["Action/MeanAction"])
				astd.append(row["Action/StdAction"])
				rmedian.append(row["Return/MedianReturn"])
				rlq.append(row["Return/LowerQuartileReturn"])
				ruq.append(row["Return/UpperQuartileReturn"])
				rmean.append(row["Evaluation/AverageReturn"])

			ameans.append(amean)
			astds.append(astd)
			rmedians.append(rmedian)
			rlqs.append(rlq)
			ruqs.append(ruq)
			rmeans.append(rmean)
	
	np.savez_compressed(
		dest, ameans=ameans, astds=astds, rmedians=rmedians, rlqs=rlqs,
		ruqs=ruqs, rmeans=rmeans
	)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="RL results parser")
	parser.add_argument("--fpaths", nargs="?", default="", type=str)
	parser.add_argument("--dest", default="results.npz", type=str)

	args = parser.parse_args()
	main(args.fpaths, args.dest)