import csv
import argparse
import numpy as np


def str_to_np(arr_str):
	# construct a numpy array from its string representation
	arr_str = ''.join([
		'[',
		','.join(arr_str.strip("[]").strip().split()),
		']'
	])
	return np.array(eval(arr_str))

def main(fpaths, dest):
	fpaths = fpaths.split(",")
	ameans = []
	astds = []
	rmedians = []
	rlqs = []
	ruqs = []
	emedians = []
	elqs = []
	euqs = []
	rmeans = []
	rstds = []
	rmaxs = []
	rmins = []
	pstds = []
	pmeans=[]
	emeans = []
	estds = []
	alphas = []
	qlosses = []
	plosses = []

	for fpath in fpaths:
		with open(fpath, newline='') as csvfile:
			reader = csv.DictReader(csvfile)
			amean = []
			astd = []
			rmedian = []
			rlq = []
			ruq = []
			emedian = []
			elq = []
			euq = []
			rmean = []
			rstd = []
			rmax = []
			rmin = []
			pstd = []
			pmean = []
			emean = []
			estd = []
			alpha = []
			qloss = []
			ploss = []
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
				if "Policy/MeanStd" in row: pstd.append(str_to_np(row["Policy/MeanStd"]))
				if "Policy/Mean" in row: pmean.append(str_to_np(row["Policy/Mean"]))
				if "Evaluation/AverageReturn" in row: emean.append(row["Evaluation/AverageReturn"])
				if "Evaluation/StdReturn" in row: estd.append(row["Evaluation/StdReturn"])
				if "Evaluation/MedianReturn" in row: emedian.append(row["Evaluation/MedianReturn"])
				if "Evaluation/LowerQuartileReturn" in row: elq.append(row["Evaluation/LowerQuartileReturn"])
				if "Evaluation/UpperQuartileReturn" in row: euq.append(row["Evaluation/UpperQuartileReturn"])
				if "AlphaTemperature/mean" in row: alpha.append(row["AlphaTemperature/mean"])
				if "QF/Qf1Loss" in row: qloss.append(row["QF/Qf1Loss"])
				if "AdaptiveGaussianMLPPolicy/LossAfter" in row: ploss.append(row["AdaptiveGaussianMLPPolicy/LossAfter"])

			ameans.append(amean)
			astds.append(astd)
			rmedians.append(rmedian)
			rlqs.append(rlq)
			ruqs.append(ruq)
			rmeans.append(rmean)
			emedians.append(emedian)
			elqs.append(elq)
			euqs.append(euq)
			rstds.append(rstd)
			rmaxs.append(rmax)
			rmins.append(rmin)
			pstds.append(pstd)
			pmeans.append(pmean)
			emeans.append(emean)
			estds.append(estd)
			alphas.append(alpha)
			qlosses.append(qloss)
			plosses.append(ploss)
	np.savez_compressed(
		dest,
		ameans=ameans, astds=astds, rmedians=rmedians, rlqs=rlqs, ruqs=ruqs,
		emedians=emedians, elqs=elqs, euqs=euqs, rmeans=rmeans, rstds=rstds,
		rmaxs=rmaxs, rmins=rmins, pstds=pstds, pmeans=pmeans, emeans=emeans,
		estds=estds, alphas=alphas, qlosses=qlosses, plosses=plosses
	)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="RL results parser")
	parser.add_argument("--fpaths", nargs="?", default="", type=str)
	parser.add_argument("--dest", default="results.npz", type=str)

	args = parser.parse_args()
	main(args.fpaths, args.dest)