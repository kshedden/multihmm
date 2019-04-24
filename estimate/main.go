package main

import (
	"flag"
	"io"
	"log"
	"os"

	"github.com/kshedden/multihmm/hmmlib"
)

var (
	logger *log.Logger
)

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
	flag.Parse()

	if *gobname == "" {
		_, _ = io.WriteString(os.Stderr, "'gobfile' is a required argument")
		os.Exit(1)
	}

	hmm := hmmlib.ReadHMM(*gobname)
	logger = hmm.SetLogger(*logname)
	hmm.Initialize()

	hmm.WriteOracleSummary(nil)

	// Fit the model parameters
	hmm.SetStartParams()
	hmm.WriteSummary(nil, "Starting values:")
	hmm.Fit(*maxiter)
	hmm.WriteSummary(nil, "Estimated parameters:")

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
