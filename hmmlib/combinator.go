package hmmlib

import (
	"encoding/binary"
	"hash"
	"hash/fnv"
	"math/rand"
	"sort"
)

type combinator struct {

	// scores[p] contains an ascending sequence of numbers for particle p
	// that quantify how well placing particle p in a given state fits the
	// data.  Internally to this value, scores[p][j] is the score for
	// placing particle p in state j, but externally to this class, state
	// j is mapped to a different actual state.
	scores [][]float64

	// topcap contains caps that bound the values that are produced by
	// deterministic enumeration.  Particle p is never placed in state
	// topcap[p] or higher by deterministic enumeration.
	topcap []int

	// constraint is a function that returns 0 if the constraint is met and
	// a positive number otherwise
	constraint func([]int, []bool) float64

	// hash is used for quickly determining if two slices are equal
	hash hash.Hash64

	// mask[p] is true indicates that particle p should be omitted from
	// all calculations
	mask []bool

	// seen[hashix[x]] is true iff the slic x has already been included in
	// the enumeration
	seen map[uint64]bool

	// Workspace for project
	prjwk []int
}

type crecv []combiRec

func (a crecv) Len() int           { return len(a) }
func (a crecv) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a crecv) Less(i, j int) bool { return a[i].score < a[j].score }

// combiRec represents a single set of state assignments with its score
type combiRec struct {
	score float64
	ix    []int
}

// hashix creates an integer hash value from the given slice.
func (combi *combinator) hashix(ix []int) uint64 {

	combi.hash.Reset()
	for p, i := range ix {
		if !combi.mask[p] {
			err := binary.Write(combi.hash, binary.LittleEndian, uint64(i))
			if err != nil {
				panic(err)
			}
		}
	}

	return combi.hash.Sum64()
}

// getScore returns the negative likelihood score for the given set of
// assignments.  Smaller scores correspond to greater likelihood.
func (combi *combinator) getScore(ix []int) float64 {

	var v float64
	for p, j := range ix {
		if !combi.mask[p] {
			v += combi.scores[p][j]
		}
	}

	return v
}

// NewCombinator returns a newly allocated combinator object, for the given parameters.
func NewCombinator(obspr [][]float64, constraint func([]int, []bool) float64, mask []bool) *combinator {

	return &combinator{
		scores:     obspr,
		constraint: constraint,
		hash:       fnv.New64(),
		mask:       mask,
		prjwk:      make([]int, 0, len(mask)),
	}
}

// copy creates a copy of the slice
func (combi *combinator) copy(ix []int) []int {
	ixc := make([]int, len(ix))
	copy(ixc, ix)
	return ixc
}

// project takes a slice and randomly projects it to
// a nearby slice that satisfies the constraint.
func (combi *combinator) project(ix, caps []int) bool {

	nstate := len(combi.scores[0])

	// Get indices of non-masked particles
	prjwk := combi.prjwk[0:0]
	for i, m := range combi.mask {
		if !m {
			prjwk = append(prjwk, i)
		}
	}

	for iter := 0; iter < 5000; iter++ {

		// Make a random move
		q := prjwk[rand.Int()%len(prjwk)]

		if ix[q] < combi.topcap[q] {
			ix[q] = combi.topcap[q]
		} else {
			ix[q]++
		}

		if ix[q] >= nstate {
			ix[q] = 0
		}

		if combi.constraint(ix, combi.mask) > 0 {
			continue
		}

		ha := combi.hashix(ix)
		if !combi.seen[ha] {
			combi.seen[ha] = true
			return true
		}
	}

	return false
}

func (combi *combinator) enumerateWithFixedState(caps []int, fixpos, fixval int) []combiRec {

	var rv []combiRec
	state := make([]int, len(caps))
	state[fixpos] = fixval

	for {
		// Handle the current state vector
		ix := combi.copy(state)
		if combi.constraint(state, combi.mask) == 0 {
			// The constraint is satisfied
			rv = append(rv, combiRec{combi.getScore(ix), ix})
			h := combi.hashix(ix)
			combi.seen[h] = true
		} else {
			// The constraint is not satisfied
			if combi.project(ix, caps) {
				rv = append(rv, combiRec{combi.getScore(ix), ix})
			}
		}

		// Try to advance the state
		done := true
		for j := range caps {
			if j != fixpos {
				if state[j] < caps[j]-1 {
					state[j]++
					done = false
					break
				} else {
					state[j] = 0
				}
			}
		}

		if done {
			// We reached the zero state, which means that all the
			// states have been visited.
			break
		}
	}

	sort.Sort(crecv(rv))
	return rv
}

// enumerate produces an array of distinct states with relatively high
// likelihood.  The states are sorted in decreasing likelihood order.
func (combi *combinator) enumerate(caps []int) []combiRec {

	combi.topcap = make([]int, len(caps))
	copy(combi.topcap, caps)

	// Use this to avoid including the same state multiple times
	combi.seen = make(map[uint64]bool)

	h := combi.enumerateHelper(caps)
	return h
}

func (combi *combinator) enumerateHelper(caps []int) []combiRec {

	// Check validity of caps
	for j := range caps {
		if !combi.mask[j] && caps[j] < 1 {
			panic("Invalid caps")
		}
	}

	// Find a state whose range will be reduced by 1.
	var jmx, mx int
	first := true
	for j := range caps {
		if combi.mask[j] {
			continue
		}
		if first || caps[j] > mx {
			jmx = j
			mx = caps[j]
			first = false
		}
	}

	// Base case for the recursion
	if mx == 1 {
		// Only possibly valid state is zero
		ix := make([]int, len(caps))
		for j := range ix {
			ix[j] = 0
		}

		if combi.constraint(ix, combi.mask) == 0 || combi.project(ix, caps) {
			return []combiRec{{combi.getScore(ix), ix}}
		}
		return nil
	}

	caps[jmx]--
	crec1 := combi.enumerateHelper(caps)
	crec2 := combi.enumerateWithFixedState(caps, jmx, mx-1)
	caps[jmx]++

	// Merge
	crec3 := make([]combiRec, len(crec1)+len(crec2))
	for i := 0; len(crec1) > 0 || len(crec2) > 0; i++ {
		switch {
		case len(crec1) == 0:
			crec3[i] = crec2[0]
			crec2 = crec2[1:]
		case len(crec2) == 0 || crec1[0].score < crec2[0].score:
			crec3[i] = crec1[0]
			crec1 = crec1[1:]
		default:
			crec3[i] = crec2[0]
			crec2 = crec2[1:]
		}
	}

	return crec3
}
