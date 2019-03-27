package main

import (
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"os"

	"github.com/kshedden/multihmm/hmmlib"
)

func getHMM(fname string) *hmmlib.MultiHMM {

	fid, err := os.Open(fname)
	if err != nil {
		panic(err)
	}
	defer fid.Close()

	gid, err := gzip.NewReader(fid)
	if err != nil {
		panic(err)
	}
	defer gid.Close()

	dec := gob.NewDecoder(gid)

	var hmm hmmlib.MultiHMM
	err = dec.Decode(&hmm)
	if err != nil {
		panic(err)
	}

	return &hmm
}

func sumNotEqual(x, y []int) (int, int) {

	if len(x) != len(y) {
		msg := fmt.Sprintf("%d != %d\n", len(x), len(y))
		print(msg)
		panic("Lengths are not equal")
	}

	var e, n int
	for t := range x {
		if x[t] == hmmlib.NullState && y[t] != hmmlib.NullState {
			fmt.Printf("%d %v %v\n", t, x[t], y[t])
			panic("inconsistent")
		}
		if x[t] != hmmlib.NullState && y[t] == hmmlib.NullState {
			fmt.Printf("%d %v %v\n", t, x[t], y[t])
			panic("inconsistent")
		}
		if x[t] == hmmlib.NullState {
			continue
		}
		if x[t] != y[t] {
			e++
		}
		n++
	}

	return e, n
}

func report(hmm *hmmlib.MultiHMM) int {

	var t, tn int
	for p := 0; p < hmm.NParticle; p++ {
		q, n := sumNotEqual(hmm.PState[p], hmm.State[p])
		fmt.Printf("%d/%d\n", q, n)
		t += q
		tn += n
	}
	fmt.Printf("%d/%d total errors\n", t, tn)

	return t
}

func main() {

	hmm := getHMM("tmp.gob.gz")
	hmm.VarForm = hmmlib.VarConst
	hmm.InitializeMulti()

	fid, err := os.Create("results.txt")
	if err != nil {
		panic(err)
	}
	defer fid.Close()

	hmm.WriteOracleSummary(nil, fid)

	hmm.SetStartParams()
	hmm.WriteSummary(nil, "Starting values:", fid)
	hmm.Fit(30)
	hmm.WriteSummary(nil, "Estimated parameters:", fid)
	hmm.ReconstructStates()

	// Save the standard prediction
	pstate0 := make([][]int, hmm.NParticle)
	for p := 0; p < hmm.NParticle; p++ {
		pstate0[p] = make([]int, hmm.NTime)
		copy(pstate0[p], hmm.PState[p])
	}

	fmt.Printf("\nStandard reconstruction:\n")
	report(hmm)
	hmm.ReconstructMulti()
	fmt.Printf("\nCollision-avoiding reconstruction:\n")
	report(hmm)
}
