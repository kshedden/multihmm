package main

import (
	"compress/gzip"
	"encoding/gob"
	"flag"
	"math/rand"
	"os"
	"time"

	"github.com/kshedden/multihmm/hmmlib"
)

func main() {

	var obsmodel, varmodel, outname string
	flag.StringVar(&obsmodel, "obsmodel", "gaussian", "Observation distribution")
	flag.StringVar(&varmodel, "varmodel", "constant", "Variance model")
	flag.StringVar(&outname, "outname", "", "Output file name prefix")

	var zeroinflated bool
	flag.BoolVar(&zeroinflated, "zeroinflated", false, "Zero inflated")

	var varpower float64
	flag.Float64Var(&varpower, "varpower", 1.5, "Power variance for Tweedie")

	var nParticleGrp, nState, nTime, nGroup int
	flag.IntVar(&nParticleGrp, "nparticlegrp", 0, "Number of particles per group")
	flag.IntVar(&nState, "nstate", 0, "Number of states")
	flag.IntVar(&nTime, "ntime", 0, "Number of time points")
	flag.IntVar(&nGroup, "ngroup", 0, "Number of groups")
	flag.Parse()

	rand.Seed(time.Now().UTC().UnixNano())

	hmm := hmmlib.NewMulti(nParticleGrp*nGroup, nState, nTime)

	switch obsmodel {
	case "gaussian":
		hmm.ObsModelForm = hmmlib.Gaussian
	case "poisson":
		hmm.ObsModelForm = hmmlib.Poisson
	case "tweedie":
		hmm.ObsModelForm = hmmlib.Tweedie
		hmm.VarPower = varpower
	}

	switch varmodel {
	case "const":
		hmm.VarForm = hmmlib.VarConst
	case "free":
		hmm.VarForm = hmmlib.VarFree
	}

	hmm.ZeroInflated = zeroinflated

	if outname == "" {
		panic("'outname' is required")
	}

	// Put everyone into a group
	hmm.Group = make([][]int, nGroup)
	ii := 0
	for j := 0; j < nGroup; j++ {
		for k := 0; k < nParticleGrp; k++ {
			hmm.Group[j] = append(hmm.Group[j], ii)
			ii++
		}
	}

	// Set the transition matrix
	hmm.Trans = make([]float64, nState*nState)
	for i := 0; i < nState; i++ {
		p := 0.8 + 0.1*float64(i)/float64(nState-1)
		for j := 0; j < nState; j++ {
			if i == j {
				hmm.Trans[i*nState+j] = p
			} else {
				hmm.Trans[i*nState+j] = (1 - p) / float64(nState-1)
			}
		}
	}

	// Set the initial state probabilities
	hmm.Init = make([]float64, nState)
	for i := 0; i < hmm.NState; i++ {
		hmm.Init[i] = 1 / float64(nState)
	}

	// Set the parameters of the observation distribution
	hmm.Mean = make([]float64, nState*nState)
	for i := 0; i < nState; i++ {
		for j := 0; j < nState; j++ {
			switch hmm.ObsModelForm {
			case hmmlib.Gaussian:
				if i == j {
					hmm.Mean[i*nState+j] = 1
				}
			case hmmlib.Poisson:
				if i == j {
					hmm.Mean[i*nState+j] = 8
				} else {
					hmm.Mean[i*nState+j] = 1
				}
			case hmmlib.Tweedie:
				if i == j {
					hmm.Mean[i*nState+j] = 8
				} else {
					hmm.Mean[i*nState+j] = 1
				}
			default:
				panic("unkown obsmodel\n")
			}
		}
	}

	// Set the zero inflation probabilities if needed
	if hmm.ZeroInflated {
		hmm.Zprob = make([]float64, nState*nState)
		ii := 0
		for i := 0; i < hmm.NState; i++ {
			for j := 0; j < hmm.NState; j++ {
				if i == j {
					hmm.Zprob[ii] = 0.1
				} else {
					hmm.Zprob[ii] = 0.4
				}
				ii++
			}
		}
	}

	// Set the standard deviations if needed
	if hmm.ObsModelForm == hmmlib.Gaussian {
		hmm.Std = make([]float64, nState*nState)
		for i := 0; i < nState*nState; i++ {
			hmm.Std[i] = 0.22
		}
	}

	hmm.GenStatesMulti()
	hmm.GenObs()

	fid, err := os.Create("tmp.gob.gz")
	if err != nil {
		panic(err)
	}
	defer fid.Close()

	gid := gzip.NewWriter(fid)
	defer gid.Close()

	enc := gob.NewEncoder(gid)

	if err := enc.Encode(&hmm); err != nil {
		panic(err)
	}
}
