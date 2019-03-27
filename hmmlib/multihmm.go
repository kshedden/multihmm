package hmmlib

import (
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"sort"

	"github.com/schollz/progressbar"
	"gonum.org/v1/gonum/floats"
)

const (
	// Keep this many top-scoring configurations in the joint reconstruction
	nkp = 50
)

// MultiHMM is an HMM that can do joint Viterbi reconstruction of the state sequences.
type MultiHMM struct {
	HMM

	// The forward probabilities
	fpr []float64

	// Traceback pointers
	tbp []int

	// The actual state sequence (for all the particles) at each traceback point
	lpx [][]int

	// Number of valid states for each time point
	npt []int

	// Workspace
	cwk []int
}

// NewMulti returns a MultiHMM value with the given size parameters.
func NewMulti(NParticle, NState, NTime int) *MultiHMM {

	hmm := New(NParticle, NState, NTime)

	return &MultiHMM{
		HMM: *hmm,
	}
}

// InitializeMulti allocates workspaces for parameter estimation, so
// that the multi-state Viterbi reconstruction can be applied.
func (hmm *MultiHMM) InitializeMulti() {

	if nkp > int(math.Pow(float64(hmm.NState), float64(hmm.NParticle))) {
		panic("nkp is too large for NState and NParticle")
	}

	hmm.Initialize()

	hmm.fpr = make([]float64, hmm.NTime*nkp)
	hmm.lpx = make([][]int, hmm.NTime*nkp)
	hmm.tbp = make([]int, hmm.NTime*nkp)
	hmm.npt = make([]int, hmm.NTime)

	hmm.cwk = make([]int, hmm.NParticle)
}

// Get all the observation probabilities at time t, sort them, and remove the states
// that have low probability.
func (hmm *MultiHMM) getMultiObsProb(t int, obspr [][]float64, inds [][]int, mask []bool) {

	// Get the negative observation probabilities and sort them.  The first probability
	// for each particle is always valid.
	for p := 0; p < hmm.NParticle; p++ {

		mask[p] = hmm.Obs[p][t*hmm.NState] == NullObs
		if mask[p] {
			continue
		}

		for st := 0; st < hmm.NState; st++ {
			obspr[p][st] = -hmm.GetLogObsProb(p, t, st)
		}
		floats.Argsort(obspr[p], inds[p])
	}
}

// GenStatesMulti creates a random state sequence in which
// there are no collisions.
func (hmm *MultiHMM) GenStatesMulti() {

	ntime := hmm.NTime
	hmm.State = makeIntArray(hmm.NParticle, hmm.NTime)
	row := make([]float64, hmm.NState)

	// Set the initial state
	check := make(map[int]bool)
	for p := 0; p < hmm.NParticle; p++ {
		for {
			st := genDiscrete(hmm.Init)
			if !check[st] {
				hmm.State[p][0] = st
				check[st] = true
				break
			}
		}
	}

	// Set the rest of the states
	for t := 1; t < ntime; t++ {
		check = make(map[int]bool)
		for p := 0; p < hmm.NParticle; p++ {
			st0 := hmm.State[p][t-1]
			copy(row, hmm.Trans[st0*hmm.NState:(st0+1)*hmm.NState])
			for {
				st := genDiscrete(row)
				if !check[st] {
					hmm.State[p][t] = st
					check[st] = true
					break
				}
			}
		}
	}

	// Mask
	for p := 0; p < hmm.NParticle; p++ {
		entry := rand.Int() % 100
		for t := 0; t < entry; t++ {
			hmm.State[p][t] = NullState
		}

		exit := 900 + (rand.Int() % 100)
		for t := exit; t < hmm.NTime; t++ {
			hmm.State[p][t] = NullState
		}
	}
}

// getCaps finds the lowest set of caps that provide at least nkp
// state combinations.
func (hmm *MultiHMM) getCaps(scores [][]float64, nkp int) []int {

	// Initial caps
	caps := make([]int, hmm.NParticle)
	for j := range caps {
		caps[j] = 1
	}

	// size returns the number of state combinations that fall below the cap.
	size := func(caps []int) int {
		s := 1
		for j := range caps {
			s *= caps[j]
		}
		return s
	}

	for size(caps) < nkp {
		lm := 0.0 // Value of lowest cap
		lj := 0   // Position of cap to raise
		first := true
		for p := range caps {
			if caps[p] < hmm.NState {

				// Should this be relative or absolute?
				z := scores[p][caps[p]] - scores[p][0]
				if first || z < lm {
					lm = z
					lj = p
					first = false
				}
			}
		}
		if first {
			// Can't create enough states
			break
		}
		// Raise the cap
		caps[lj]++
	}

	return caps
}

// getValid constructs an array of valid multistates in ascending score order for
// time point t.  On exit, obspr will hold the negative observation probabilities.
func (hmm *MultiHMM) getValid(t int, obspr [][]float64, inds [][]int, mask []bool) []combiRec {

	// Make a constraint function that returns true if no two particles
	// are in the same state.
	constraint := func(ix []int, mask []bool) float64 {

		wk := hmm.cwk
		wk = wk[0:0]
		for p, j := range ix {
			if !mask[p] {
				wk = append(wk, inds[p][j])
			}
		}
		sort.IntSlice(wk).Sort()

		v := 0
		for j := 1; j < len(wk); j++ {
			if wk[j] == wk[j-1] {
				v++
			}
		}

		return float64(v)
	}

	// Gradually raise the caps until we get enough points.
	var ipa []combiRec
	f := 1
	for {
		hmm.getMultiObsProb(t, obspr, inds, mask)

		// If all the particles are masked, there are no states to return
		ms := 0
		for j := range mask {
			if !mask[j] {
				ms++
			}
		}
		if ms == 0 {
			return nil
		}

		combi := &combinator{
			scores:     obspr,
			constraint: constraint,
			hash:       fnv.New64(),
			mask:       mask,
		}

		caps := hmm.getCaps(obspr, f*nkp)
		ipa = combi.enumerate(caps)

		if len(ipa) >= nkp {
			break
		}

		// If we didn't get enough valid states, try again by retreiving more states.
		f *= 2
		if f > 10000 {
			fmt.Printf("having trouble finding states that satisfy the constraint...\n")
			fmt.Printf("t=%d, len(ipa)=%d, nkp=%d\n", t, len(ipa), nkp)
		}
	}

	if len(ipa) > nkp {
		ipa = ipa[0:nkp]
	}

	// The ipa values are position in the sorted probability lists.  Here we
	// convert them to the actual state codes.
	hmm.recodeToStates(ipa, inds)

	return ipa
}

// recodeToStates converts positions to state values
func (hmm *MultiHMM) recodeToStates(ipa []combiRec, inds [][]int) {
	for _, x := range ipa {
		for p := range x.ix {
			x.ix[p] = inds[p][x.ix[p]]
		}
	}
}

// multiTrans returns the log of the joint transition probability
// from multistate ix1 to multistate ix2.
func (hmm *MultiHMM) multiTrans(states1, states2 []int, mask []bool) float64 {

	fpr := 0.0
	for j := 0; j < hmm.NParticle; j++ {
		if !mask[j] {
			st1, st2 := states1[j], states2[j]
			fpr += math.Log(hmm.Trans[st1*hmm.NState+st2])
		}
	}

	return fpr
}

// ReonstructMulti uses a modified Viterbi algorithm to predict
// the latent state sequence in a way that satistifes the constraints.
func (hmm *MultiHMM) ReconstructMulti() {

	hmm.PState = makeIntArray(hmm.NParticle, hmm.NTime)

	hmm.multiprob()
	hmm.traceback()
}

// traceback is the Viterbi traceback
func (hmm *MultiHMM) traceback() {

	fpr := hmm.fpr
	tbp := hmm.tbp
	lpx := hmm.lpx
	npt := hmm.npt
	pstate := hmm.PState
	multistate := make([]int, hmm.NTime)

	t := hmm.NTime
	jt := nkp * t

	// Loop over blocks separated by intervals with no data
	for t > 0 {

		t--
		jt -= nkp

		// Keep backing up until we reach a point with data
		for t >= 0 && npt[t] == 0 {
			multistate[t] = NullState
			t--
			jt -= nkp
		}
		if t < 0 {
			// We backed up all the way to the beginning
			break
		}

		// Find the best final state
		qpr := fpr[jt]
		bst := 0
		for j := 1; j < npt[t]; j++ {
			if fpr[jt+j] < qpr {
				qpr = fpr[jt+j]
				bst = j
			}
		}
		multistate[t] = bst
		t--
		jt -= nkp

		// Trace back until we get to a point with no data
		for t >= 0 && npt[t] > 0 {
			multistate[t] = tbp[jt+nkp+multistate[t+1]]
			t--
			jt -= nkp
		}
	}

	// Fill in PState
	jt = 0
	for t := 0; t < hmm.NTime; t++ {
		if multistate[t] != NullState {
			ix := lpx[jt+multistate[t]]
			for p := 0; p < hmm.NParticle; p++ {
				if hmm.Obs[p][t*hmm.NState] == NullObs {
					pstate[p][t] = NullState
				} else {
					pstate[p][t] = ix[p]
				}
			}
		} else {
			for p := 0; p < hmm.NParticle; p++ {
				pstate[p][t] = NullState
			}
		}
		jt += nkp
	}
}

// multiprob calculates the forward chain probabilities used by the
// Viterbi traceback.
func (hmm *MultiHMM) multiprob() {

	print("Jointly predicting state sequences...\n")
	bar := progressbar.New(hmm.NTime)

	fpr := hmm.fpr
	lpx := hmm.lpx
	tbp := hmm.tbp
	npt := hmm.npt

	mask := make([]bool, hmm.NParticle)
	obspr := makeFloatArray(hmm.NParticle, hmm.NState)
	inds := makeIntArray(hmm.NParticle, hmm.NState)

	// Calculate forward probabilities
	j0 := -2 * nkp
	jt := -nkp
	for t := 0; t < hmm.NTime; t++ {

		j0 += nkp
		jt += nkp

		bar.Add(1)

		ipa := hmm.getValid(t, obspr, inds, mask)
		npt[t] = len(ipa)

		if npt[t] == 0 {
			continue
		}

		// Starting over
		if t == 0 || npt[t-1] == 0 {

			for jj, cr := range ipa {

				lpg := 0.0
				for p, st := range cr.ix {
					if !mask[p] {
						lpg += -math.Log(hmm.Init[st]) - hmm.GetLogObsProb(p, t, st)
					}
				}

				fpr[jt+jj] = lpg
				lpx[jt+jj] = cr.ix
				// no tbp
			}
			continue
		}

		for jj, cr := range ipa {

			var lpu float64
			var ipu int
			for j := 0; j < npt[t-1]; j++ {
				lx := fpr[j0+j] - hmm.multiTrans(lpx[j0+j], cr.ix, mask)
				if j == 0 || lx < lpu {
					lpu = lx
					ipu = j
				}
			}

			ltu := 0.0
			for p, st := range cr.ix {
				// Can't use obspr here because it has been sorted
				if !mask[p] {
					ltu -= hmm.GetLogObsProb(p, t, st)
				}
			}

			fpr[jt+jj] = lpu + ltu
			tbp[jt+jj] = ipu
			lpx[jt+jj] = cr.ix
		}
	}
}

func resizeInt(x []int, m int) []int {
	if cap(x) >= m {
		return x[0:m]
	}
	return make([]int, m)
}

func resizeFloat(x []float64, m int) []float64 {
	if cap(x) >= m {
		return x[0:m]
	}
	return make([]float64, m)
}

func resizeInt2(x [][]int, m int) [][]int {
	if cap(x) >= m {
		return x[0:m]
	}
	return make([][]int, m)
}
