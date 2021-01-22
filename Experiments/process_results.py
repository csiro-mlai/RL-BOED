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
	rstds = []
	rmaxs = []
	rmins = []
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
			rstd = []
			rmax = []
			rmin = []
			pstd = []
			for row in reader:
				if "Action/MeanAction" in row: amean.append(row["Action/MeanAction"])
				if "Action/StdAction" in row: astd.append(row["Action/StdAction"])
				if "Return/MeanReturn" in row: rmean.append(row["Return/MeanReturn"])
				if "Return/StdReturn" in row: rstd.append(row["Return/StdReturn"])
				if "Return/MaxReturn" in row: rmax.append(row["Return/MaxReturn"])
				if "Return/MinReturn" in row: rmin.append(row["Return/MinReturn"])
				if "Return/MedianReturn" in row: rmedian.append(row["Return/MedianReturn"])
				if "Return/LowerQuartileReturn" in row: rlq.append(row["Return/LowerQuartileReturn"])
				if "Return/UpperQuartileReturn" in row: ruq.append(row["Return/UpperQuartileReturn"])
				if "Policy/MeanStd" in row: pstd.append(row["Policy/MeanStd"])

			ameans.append(amean)
			astds.append(astd)
			rmedians.append(rmedian)
			rlqs.append(rlq)
			ruqs.append(ruq)
			rmeans.append(rmean)
			rstds.append(rstd)
			rmaxs.append(rmax)
			rmins.append(rmin)
			pstds.append(pstd)
	
	np.savez_compressed(
		"results.npz",
		ameans=ameans, astds=astds, rmedians=rmedians, rlqs=rlqs, ruqs=ruqs,
		rmeans=rmeans, rstds=rstds, rmaxs=rmaxs, rmins=rmins, pstds=pstds, 
	)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="RL results parser")
	parser.add_argument("--fpaths", nargs="?", default="", type=str)
	parser.add_argument("--dest", default="results.npz", type=str)

	args = parser.parse_args()
	main(args.fpaths, args.dest)