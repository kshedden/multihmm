package hmmlib

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"

	"github.com/schollz/progressbar"
	"gonum.org/v1/gonum/floats"
)

// MultiHMM is an HMM that can do joint Viterbi reconstruction of the state sequences.
type MultiHMM struct {
	HMM

	// Group[k] lists all the particles that are reconstructed jointly
	Group [][]int
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
func (hmm *MultiHMM) InitializeMulti(nkp int) {

	if math.Log(float64(nkp)) > float64(hmm.NParticle)*math.Log(float64(hmm.NState)) {
		msg := fmt.Sprintf("nkp (%d) is too large for NState (%d) and NParticle (%d)", nkp,
			hmm.NState, hmm.NParticle)
		panic(msg)
	}

	hmm.msglogger.Printf("Group sizes:")
	var mx int
	for p, og := range hmm.Group {
		if len(og) > mx {
			mx = len(og)
		}
		hmm.msglogger.Printf("%4d  %8d\n", p, len(og))
	}

	if float64(mx) >= 0.9*float64(hmm.NState) {
		msg := "There are too many particles to do a collision-avoiding fit."
		_, _ = io.WriteString(os.Stderr, msg)
	}
}

// Get all the observation probabilities at time t, sort them, and remove the states
// that have low probability.
func (hmm *MultiHMM) getMultiObsProb(t int, obsgrp []int, obspr [][]float64, inds [][]int, mask []bool) {

	// Get the negative observation probabilities and sort them.
	for j, p := range obsgrp {

		mask[j] = hmm.Obs[p][t*hmm.NState] == NullObs
		if mask[j] {
			continue
		}

		fprob := hmm.Fprob[p][t*hmm.NState : (t+1)*hmm.NState]
		bprob := hmm.Bprob[p][t*hmm.NState : (t+1)*hmm.NState]
		floats.MulTo(obspr[j], fprob, bprob)
		normalizeSum(obspr[j], 0)
		for i := range obspr[j] {
			obspr[j][i] = -math.Log(obspr[j][i])
		}

		floats.Argsort(obspr[j], inds[j])
	}
}

// GenStatesMulti creates a random state sequence in which
// there are no collisions.
func (hmm *MultiHMM) GenStatesMulti() {

	ntime := hmm.NTime
	hmm.State = makeIntArray(hmm.NParticle, hmm.NTime)
	row := make([]float64, hmm.NState)

	// Set the initial state
	for p := 0; p < hmm.NParticle; p++ {
		hmm.State[p][0] = genDiscrete(hmm.Init)
	}

	// Set the rest of the states
	for t := 1; t < ntime; t++ {

		for g := range hmm.Group {

			// Try to avoid collisions within a group at the same time
			check := make(map[int]bool)

			for _, p := range hmm.Group[g] {

				st0 := hmm.State[p][t-1]
				copy(row, hmm.Trans[st0*hmm.NState:(st0+1)*hmm.NState])

				for k := 0; ; k++ {
					st := genDiscrete(row)

					// Try 100 times to avoid a collision
					if k < 100 && check[st] {
						continue
					}

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
func (hmm *MultiHMM) getCaps(scores [][]float64, mask []bool, nkp int) []int {

	// Initial caps
	caps := make([]int, len(scores))
	for j := range caps {
		if !mask[j] {
			caps[j] = 1
		}
	}

	// size returns the number of state combinations that fall below the cap.
	size := func(caps []int) int {
		s := 1
		for j := range caps {
			if !mask[j] {
				s *= caps[j]
			}
		}
		return s
	}

	for size(caps) < nkp {
		lm := 0.0 // Value of lowest cap
		lj := 0   // Position of cap to raise
		first := true
		for p := range caps {
			if mask[p] {
				continue
			}
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
func (hmm *MultiHMM) getValid(t int, obsgrp []int, obspr [][]float64, inds [][]int, mask []bool, nkp int) []combiRec {

	// obspr, inds, and mask are set here
	hmm.getMultiObsProb(t, obsgrp, obspr, inds, mask)

	// Get the number of unmasked particles
	ump := 0
	for i := range mask {
		if !mask[i] {
			ump++
		}
	}

	// Everything in the group is masked
	if ump == 0 {
		return nil
	}

	// Get the maximum number of states.  In some cases it may be impossible
	// to find nkp multistates, so adjust accordingly.
	if math.Log(float64(nkp)) > float64(ump)*math.Log(float64(hmm.NState)) {
		nkp = int(math.Pow(float64(hmm.NState), float64(ump)))
	}

	nparticle := len(obsgrp)

	// Workspace for constraint function.
	wk := make([]int, nparticle)

	// Make a constraint function that returns true if no two particles
	// are in the same state.
	constraint := func(ix []int, mask []bool) float64 {

		wk = wk[0:0]
		for pj, sj := range ix {
			if !mask[pj] {
				wk = append(wk, inds[pj][sj])
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
	for f := 1; f*nkp < 2000; f *= 2 {
		combi := NewCombinator(obspr, constraint, mask)
		caps := hmm.getCaps(obspr, mask, f*nkp)
		ipa = combi.enumerate(caps)

		if len(ipa) >= nkp {
			break
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
	for j := range mask {
		if !mask[j] {
			st1, st2 := states1[j], states2[j]
			fpr += math.Log(hmm.Trans[st1*hmm.NState+st2])
		}
	}

	return fpr
}

// workspace for multi-state reconstruction
type rws struct {
	fpr []float64
	lpx [][]int
	tbp []int
	npt []int
}

func newRws(ntime, nkp int) *rws {
	return &rws{
		fpr: make([]float64, ntime*nkp),
		lpx: make([][]int, ntime*nkp),
		tbp: make([]int, ntime*nkp),
		npt: make([]int, ntime*nkp),
	}
}

// ReonstructMulti uses a modified Viterbi algorithm to predict
// the latent state sequence in a way that satistifes the constraints.
func (hmm *MultiHMM) ReconstructMulti(nkp int) {

	fmt.Printf("\nJointly predicting state sequences...\n")
	bar := progressbar.New(len(hmm.Group))

	var wg sync.WaitGroup

	for _, obsgrp := range hmm.Group {

		wg.Add(1)
		go func(obsgrp []int) {
			ws := newRws(hmm.NTime, nkp)
			hmm.multiprob(obsgrp, nkp, ws)
			hmm.traceback(obsgrp, nkp, ws)
			_ = bar.Add(1)
			wg.Done()
		}(obsgrp)
	}

	wg.Wait()

	fmt.Printf("\n") // returns the prompt in the usual place
}

// traceback is the Viterbi traceback
func (hmm *MultiHMM) traceback(obsgrp []int, nkp int, ws *rws) {

	fpr := ws.fpr
	lpx := ws.lpx
	tbp := ws.tbp
	npt := ws.npt

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

	// Fill in PState with the actual state labels
	jt = 0
	for t := 0; t < hmm.NTime; t++ {
		if multistate[t] != NullState {
			ix := lpx[jt+multistate[t]]
			for j, p := range obsgrp {
				if hmm.Obs[p][t*hmm.NState] == NullObs {
					pstate[p][t] = NullState
				} else {
					pstate[p][t] = ix[j]
				}
			}
		} else {
			for _, p := range obsgrp {
				pstate[p][t] = NullState
			}
		}
		jt += nkp
	}
}

// multiprob calculates the forward chain probabilities used by the
// Viterbi traceback.
func (hmm *MultiHMM) multiprob(obsgrp []int, nkp int, ws *rws) {

	fpr := ws.fpr
	lpx := ws.lpx
	tbp := ws.tbp
	npt := ws.npt

	nparticle := len(obsgrp)
	mask := make([]bool, nparticle)
	obspr := makeFloatArray(nparticle, hmm.NState)
	inds := makeIntArray(nparticle, hmm.NState)

	// Calculate forward probabilities
	j0 := -2 * nkp
	jt := -nkp
	for t := 0; t < hmm.NTime; t++ {

		j0 += nkp
		jt += nkp

		ipa := hmm.getValid(t, obsgrp, obspr, inds, mask, nkp)
		npt[t] = len(ipa)

		if npt[t] == 0 {
			continue
		}

		// Starting over
		if t == 0 || npt[t-1] == 0 {

			for jj, cr := range ipa {

				lpg := 0.0
				for pj, st := range cr.ix {
					if !mask[pj] {
						lpg += -math.Log(hmm.Init[st]) - hmm.GetLogObsProb(obsgrp[pj], t, st)
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
			for pj, st := range cr.ix {
				// Can't use obspr here because it has been sorted
				if !mask[pj] {
					ltu -= hmm.GetLogObsProb(obsgrp[pj], t, st)
				}
			}

			fpr[jt+jj] = lpu + ltu
			tbp[jt+jj] = ipu
			lpx[jt+jj] = cr.ix
		}
	}
}
