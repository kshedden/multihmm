package main

import (
	"flag"
	"io"
	"log"
	"os"
	"sort"

	"github.com/kshedden/multihmm/hmmlib"
)

var (
	logger *log.Logger
)

func flexCollisionConstraintMaker(inds [][]int) hmmlib.ConstraintFunc {

	p := len(inds)
	wk := make([]int, p)

	return func(ix []int, mask []bool) float64 {

		var v int

		for iq := 0; iq < 2; iq++ {

			var i1, i2 int
			if iq == 0 {
				i1, i2 = 0, p/2
			} else {
				i1, i2 = p/2, p
			}

			wk = wk[0:0]
			for i := i1; i < i2; i++ {
				if !mask[i] {
					wk = append(wk, inds[i][ix[i]])
				}
			}
			sort.IntSlice(wk).Sort()

			for j := 1; j < len(wk); j++ {
				if wk[j] == wk[j-1] {
					v++
				}
			}
		}

		return float64(v)
	}
}

func report(logger *log.Logger, hmm *hmmlib.MultiHMM) int {

	var t, tn int
	logger.Printf("Per-particle errors:")
	for p := 0; p < hmm.NParticle; p++ {
		q, n := hmmlib.CompareStates(hmm.PState[p], hmm.State[p])
		logger.Printf("%d %d/%d\n", p, q, n)
		t += q
		tn += n
	}
	logger.Printf("%d/%d total errors\n", t, tn)

	return t
}

func main() {

	gobname := flag.String("gobfile", "", "The data file")
	nkp := flag.Int("nkp", 200, "Number of joint states to retain")
	logname := flag.String("logname", "hmm", "Prefix of log file")
	maxiter := flag.Int("maxiter", 20, "Maximum number of iterations")
	constraint := flag.String("constraint", "", "Type of state constraint")
	reconstruct := flag.Bool("reconstruct", true, "If false, do not reconstruct states")
	varpower := flag.Float64("varpower", 0, "Override varpower in HMM gob file")
	flag.Parse()

	if *gobname == "" {
		_, _ = io.WriteString(os.Stderr, "'gobfile' is a required argument")
		os.Exit(1)
	}

	hmm := hmmlib.ReadHMM(*gobname)
	logger = hmm.SetLogger(*logname)

	switch *constraint {
	case "nocollision":
		hmm.ConstraintGen = hmmlib.NoCollisionConstraintMaker
		logger.Printf("Using no collision rules")
	case "flexcollision":
		hmm.ConstraintGen = flexCollisionConstraintMaker
		logger.Printf("Using flex collision rules")
	default:
		hmm.ConstraintGen = hmmlib.NoConstraintMaker
		logger.Printf("No constraints on collisions")
	}

	if *varpower != 0 {
		hmm.VarPower = *varpower
	}

	hmm.Initialize()

	hmm.WriteOracleSummary(nil)

	// Fit the model parameters
	hmm.SetStartParams()
	hmm.WriteSummary(nil, "Starting values:")
	hmm.Fit(*maxiter)
	hmm.WriteSummary(nil, "Estimated parameters:")

	logger.Printf("Final log-likelihood: %f", hmm.Loglike())

	if !*reconstruct {
		return
	}

	// Reconstruct each particle individually
	hmm.ReconstructStates()

	// Save the standard prediction
	pstate0 := make([][]int, hmm.NParticle)
	for p := 0; p < hmm.NParticle; p++ {
		pstate0[p] = make([]int, hmm.NTime)
		copy(pstate0[p], hmm.PState[p])
	}

	logger.Printf("\nStandard reconstruction:\n")
	report(logger, hmm)

	// Reconstruct jointly
	hmm.InitializeMulti(*nkp)
	hmm.ReconstructMulti(*nkp)
	logger.Printf("\nCollision-avoiding reconstruction:\n")
	report(logger, hmm)
}
