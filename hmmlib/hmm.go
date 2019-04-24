package hmmlib

import (
	"bytes"
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"sync"

	"github.com/schollz/progressbar"
	"gonum.org/v1/gonum/floats"
)

const (
	// Maximum allowed value for Zprob
	zpmax float64 = 0.95

	// The Poisson/Tweedie mean parameters are never allowed to go below this value
	minPoissonMean = 1e-2

	// Minimum allowed value for the observation SD
	sdmin = 1e-8

	// Null value for an observation.  When the observed value is Null, it is
	// not used in estimation.  If the first component of the observed vector
	// is Null, the are all treated as Null.
	NullObs float64 = -9999

	// Null value for a state
	NullState int = -9999
)

// VarModelType indicates how the variance structure is modeled.
type VarModelType uint8

const (
	VarFree VarModelType = iota
	VarConst
)

type ObsModelType uint8

const (
	Gaussian ObsModelType = iota
	Poisson
	Tweedie
)

// HMM represents a hidden Markov model for a collection of particles
// that follow the same HMM law.
type HMM struct {

	// Number of particles (e.g. subjects)
	NParticle int

	// Number of time points
	NTime int

	// Number of states
	NState int

	// The type of variance constraint
	VarForm VarModelType

	// The observation distribution
	ObsModelForm ObsModelType

	// If true the model is zero-inflated
	ZeroInflated bool

	// The transition probability matrix
	Trans []float64

	// The initial probability distribution
	Init []float64

	// The observation means. If ZeroInflated=true, these are the observation means
	// given that they are not zero.
	Mean []float64

	// The observation standard deviations.  If ZeroInflated=true, these are the
	// observation means given that they are not zero.
	Std []float64

	// The log probability of observing zero, only used if ZeroInflated=true.
	Zprob []float64

	// The true states (if known)
	State [][]int

	// The observations
	Obs [][]float64

	// The forward probabilities
	Fprob [][]float64

	// The backward probabilities
	Bprob [][]float64

	// The reconstructed states
	PState [][]int

	// The overall log-likelihood
	LLF []float64

	// The log-likelihood function for one particle
	llf []float64

	// The Tweedie variance is mean^VarPower
	VarPower float64

	// Write log messages here
	msglogger *log.Logger
	parlogger *log.Logger
}

// New returns an HMM value with the given size parameters.
func New(NParticle, NState, NTime int) *HMM {

	hmm := &HMM{
		NParticle: NParticle,
		NTime:     NTime,
		NState:    NState,
	}

	return hmm
}

// SetLog provides a logger that will be used to write logging messages.
func (hmm *HMM) SetLogger(logname string) *log.Logger {

	fid, err := os.Create(logname + "_msg.log")
	if err != nil {
		panic(err)
	}
	hmm.msglogger = log.New(fid, "", log.Ltime)

	fid, err = os.Create(logname + "_par.log")
	if err != nil {
		panic(err)
	}
	hmm.parlogger = log.New(fid, "", 0)

	// The calling program can also use this logger
	return hmm.msglogger
}

// Initialize allocates workspaces for parameter estimation.  Call
// this prior to calling Fit.
func (hmm *HMM) Initialize() {

	if hmm.ZeroInflated && hmm.ObsModelForm != Gaussian {
		fmt.Printf("When ZeroInflated is true, ObsModelForm must be Gaussian")
		panic("")
	}

	hmm.Bprob = makeFloatArray(hmm.NParticle, hmm.NState*hmm.NTime)
	hmm.Fprob = makeFloatArray(hmm.NParticle, hmm.NState*hmm.NTime)
	hmm.llf = make([]float64, hmm.NParticle)

	if hmm.msglogger == nil {
		hmm.msglogger = log.New(os.Stderr, "", log.Ltime)
	}

	hmm.msglogger.Printf("%d particles\n", hmm.NParticle)
	hmm.msglogger.Printf("%d time points per particle\n", hmm.NTime)
	hmm.msglogger.Printf("%d states\n", hmm.NState)
}

// ReconstructStates uses the Viterbi algorithm to predict
// the sequence of states for each particle.  The algorithm
// is run separately for each particle.  The reconstructed
// states are written into the PState component of the HMM.
func (hmm *HMM) ReconstructStates() {

	hmm.PState = makeIntArray(hmm.NParticle, hmm.NTime)

	for p := 0; p < hmm.NParticle; p++ {
		hmm.ReconstructParticle(p)
	}
}

// ReconstructParticle uses the Viterbi algorithm to predict
// the sequence of states for one particle.
func (hmm *HMM) ReconstructParticle(p int) {

	lpr := make([]float64, hmm.NTime*hmm.NState)
	lpt := make([]int, hmm.NTime*hmm.NState)

	hmm.reconstructionProbs(p, lpr, lpt)
	hmm.traceback(p, lpr, lpt)
}

func (hmm *HMM) reconstructionProbs(p int, lpr []float64, lpt []int) {

	obs := hmm.Obs[p]
	wk := make([]float64, hmm.NState)

	// Construct the table of conditional probabilities
	j0 := -2 * hmm.NState
	j1 := -hmm.NState
	for t := 0; t < hmm.NTime; t++ {

		j0 += hmm.NState
		j1 += hmm.NState

		if obs[j1] == NullObs {
			continue
		}

		// Beginning from initial conditions
		if t == 0 || obs[j0] == NullObs {
			for st := 0; st < hmm.NState; st++ {
				lpr[j1+st] = hmm.GetLogObsProb(p, t, st) + math.Log(hmm.Init[st])
			}
			continue // First block of lpt is not used
		}

		// From st1 to st2
		for st2 := 0; st2 < hmm.NState; st2++ {
			for st1 := 0; st1 < hmm.NState; st1++ {
				wk[st1] = lpr[j0+st1] + math.Log(hmm.Trans[st1*hmm.NState+st2])
			}

			// The best previous state
			jj := argmax(wk)
			lpt[j1+st2] = jj
			lpr[j1+st2] = wk[jj] + hmm.GetLogObsProb(p, t, st2)
		}
	}
}

func (hmm *HMM) traceback(p int, lpr []float64, lpt []int) {

	obs := hmm.Obs[p]
	y := hmm.PState[p]
	jt := hmm.NState * hmm.NTime

	for t := hmm.NTime - 1; t >= 0; t-- {

		jt -= hmm.NState

		if obs[jt] == NullObs {
			y[t] = NullState
			continue
		}

		// Starting a new block
		if t == len(y)-1 || (obs[jt] != NullObs && obs[jt+hmm.NState] == NullObs) {
			a := t * hmm.NState
			y[t] = argmax(lpr[a : a+hmm.NState])
			continue
		}

		y[t] = lpt[jt+hmm.NState+y[t+1]]
	}
}

// SetStartParams sets the starting parameters for the EM (Baum-Welch)
// optimization.
func (hmm *HMM) SetStartParams() {

	mean, std := hmm.MarginalMoments()

	hmm.Mean = make([]float64, hmm.NState*hmm.NState)
	for i := 0; i < hmm.NState; i++ {
		for j := 0; j < hmm.NState; j++ {
			if i == j {
				hmm.Mean[i*hmm.NState+j] = mean[i]
			} else {
				hmm.Mean[i*hmm.NState+j] = mean[i] / 10
			}
		}
	}

	if hmm.ObsModelForm == Gaussian {
		hmm.Std = make([]float64, hmm.NState*hmm.NState)
		sdmn := floats.Sum(std) / float64(len(std))
		for i := 0; i < hmm.NState; i++ {
			for j := 0; j < hmm.NState; j++ {
				if hmm.VarForm == VarConst {
					hmm.Std[i*hmm.NState+j] = sdmn
				} else {
					hmm.Std[i*hmm.NState+j] = std[i]
				}
			}
		}
	}

	if hmm.ZeroInflated {
		hmm.Zprob = make([]float64, hmm.NState*hmm.NState)
		for i := 0; i < hmm.NState; i++ {
			for j := 0; j < hmm.NState; j++ {
				if i == j {
					hmm.Zprob[i*hmm.NState+j] = 0.1
				} else {
					hmm.Zprob[i*hmm.NState+j] = 0.5
				}
			}
		}
	}

	hmm.Trans = make([]float64, hmm.NState*hmm.NState)
	for i := 0; i < hmm.NState; i++ {
		for j := 0; j < hmm.NState; j++ {
			if i == j {
				hmm.Trans[i*hmm.NState+j] = 0.8
			} else {
				hmm.Trans[i*hmm.NState+j] = 0.2 / float64(hmm.NState-1)
			}
		}
	}

	hmm.Init = make([]float64, hmm.NState)
	for i := 0; i < hmm.NState; i++ {
		hmm.Init[i] = 1 / float64(hmm.NState)
	}
}

// GetLogObsProb returns the probability P(obs | state=st, time=t), where
// obs is an observed (emitted) value from the HMM.  Factors depending
// only on 'obs' are omitted.
func (hmm *HMM) GetLogObsProb(p, t, st int) float64 {

	switch hmm.ObsModelForm {
	case Gaussian:
		return hmm.getGaussianLogObsProb(p, t, st)
	case Poisson:
		return hmm.getPoissonLogObsProb(p, t, st)
	case Tweedie:
		return hmm.getTweedieLogObsProb(p, t, st)
	default:
		panic("unknown model")
	}
}

func (hmm *HMM) getTweedieLogObsProb(p, t, st int) float64 {

	// http://www.statsci.org/smyth/pubs/tweediepdf-series-preprint.pdf

	obs := hmm.Obs[p]

	if obs[t*hmm.NState] == NullObs {
		panic("should not reach here")
	}

	pw := hmm.VarPower

	var lpr float64
	ii := st * hmm.NState
	for st1 := 0; st1 < hmm.NState; st1++ {
		y := obs[t*hmm.NState+st1]
		mn := hmm.Mean[ii]
		if mn < minPoissonMean {
			mn = minPoissonMean
		}
		lmn := math.Log(mn)
		lpr += y*math.Exp((1-pw)*lmn)/(1-pw) - math.Exp((2-pw)*lmn)/(2-pw)
		ii++
	}

	return lpr
}

func (hmm *HMM) getPoissonLogObsProb(p, t, st int) float64 {

	obs := hmm.Obs[p]

	if obs[t*hmm.NState] == NullObs {
		panic("should not reach here")
	}

	var lpr float64
	ii := st * hmm.NState
	for st1 := 0; st1 < hmm.NState; st1++ {
		y := obs[t*hmm.NState+st1]
		mn := hmm.Mean[ii]
		if mn < minPoissonMean {
			mn = minPoissonMean
		}
		lpr += -mn + y*math.Log(mn)
		ii++
	}

	return lpr
}

func (hmm *HMM) getGaussianLogObsProb(p, t, st int) float64 {

	obs := hmm.Obs[p]

	if obs[t*hmm.NState] == NullObs {
		panic("should not reach here")
	}

	var lpr float64
	ii := st * hmm.NState
	for st1 := 0; st1 < hmm.NState; st1++ {
		y := obs[t*hmm.NState+st1]
		mn := hmm.Mean[ii]
		sd := hmm.Std[ii]
		if hmm.ZeroInflated {
			zpr := hmm.Zprob[ii]
			if y == 0 {
				lpr += math.Log(zpr)
			} else {
				z := (y - mn) / sd
				q := -math.Log(sd) - z*z/2
				lpr += q + math.Log(1.0-zpr)
			}
		} else {
			z := (y - mn) / sd
			lpr += -math.Log(sd) - z*z/2
		}
		ii++
	}

	return lpr
}

// GenStates generates a random state sequence.  See GenStatesMulti for a version
// of this function that avoids collisions.
func (hmm *HMM) GenStates() {

	hmm.State = makeIntArray(1, hmm.NTime)

	ii := 0
	for p := 0; p < hmm.NParticle; p++ {

		// Set the initial state
		hmm.State[p][0] = genDiscrete(hmm.Init)

		// Set the rest of the states
		for t := 1; t < hmm.NTime; t++ {
			st := hmm.State[p][t-1]
			row := hmm.Trans[st*hmm.NState : (st+1)*hmm.NState]
			hmm.State[p][t] = genDiscrete(row)
		}

		ii += hmm.NTime
	}
}

// GenObs generates a random observation sequence.
func (hmm *HMM) GenObs() {

	hmm.Obs = makeFloatArray(hmm.NParticle, hmm.NState*hmm.NTime)
	for p := 0; p < hmm.NParticle; p++ {
		for t := 0; t < hmm.NTime; t++ {

			st := hmm.State[p][t]
			ip := st * hmm.NState

			if st == NullState {
				for j := 0; j < hmm.NState; j++ {
					hmm.Obs[p][t*hmm.NState+j] = NullObs
				}
			} else {
				for j := 0; j < hmm.NState; j++ {
					switch hmm.ObsModelForm {
					case Gaussian:
						if hmm.ZeroInflated && rand.Float64() < hmm.Zprob[ip+j] {
							hmm.Obs[p][t*hmm.NState+j] = 0
						} else {
							u := rand.NormFloat64()
							hmm.Obs[p][t*hmm.NState+j] = hmm.Mean[ip+j] + u*hmm.Std[ip+j]
						}
					case Poisson:
						hmm.Obs[p][t*hmm.NState+j] = genPoisson(hmm.Mean[ip+j])
					case Tweedie:
						hmm.Obs[p][t*hmm.NState+j] = genTweedie(hmm.Mean[ip+j], 1, hmm.VarPower)
					default:
						panic("unknown model\n")
					}
				}
			}
		}
	}
}

// ForwardBackward calculates the forward and backward probabilities using
// the usual recursive approach.
func (hmm *HMM) ForwardBackward() {

	var wg sync.WaitGroup

	for p := 0; p < hmm.NParticle; p++ {
		wg.Add(1)
		go hmm.ForwardParticle(p, &wg)
		wg.Add(1)
		go hmm.BackwardParticle(p, &wg)
	}

	wg.Wait()
}

// ForwardParticle calculates the forward probabilities for one particle.
func (hmm *HMM) ForwardParticle(p int, wg *sync.WaitGroup) {

	defer wg.Done()

	var llf float64
	fprob := hmm.Fprob[p]
	obs := hmm.Obs[p]

	// Due to concurrency, each particle needs its own workspace
	lt := make([]float64, hmm.NState*hmm.NState)
	terms := make([]float64, hmm.NState*hmm.NState)
	lfp := make([]float64, hmm.NState)

	// Precompute this
	for j := 0; j < hmm.NState*hmm.NState; j++ {
		lt[j] = math.Log(hmm.Trans[j])
	}

	j0 := -2 * hmm.NState // indexed of lagged observations
	j1 := -hmm.NState     // index of current observations

	// Forward sweep
	for t := 0; t < hmm.NTime; t++ {

		j0 += hmm.NState
		j1 += hmm.NState

		if obs[j1] == NullObs {
			continue
		}

		// Initial time point
		if t == 0 || obs[j0] == NullObs {
			for st := 0; st < hmm.NState; st++ {
				fprob[j1+st] = math.Log(hmm.Init[st]) + hmm.GetLogObsProb(p, t, st)
			}
			llf += normalizeMaxLog(fprob[j1 : j1+hmm.NState])
			continue
		}

		// Precompute this
		for st := 0; st < hmm.NState; st++ {
			lfp[st] = math.Log(fprob[j0+st])
		}

		// Calculate components of the update on the log scale.
		// Transition is from state st2 at time t-1 to state st1
		// at time t.
		for st1 := 0; st1 < hmm.NState; st1++ {
			yp := hmm.GetLogObsProb(p, t, st1)
			for st2 := 0; st2 < hmm.NState; st2++ {
				terms[st1*hmm.NState+st2] = lfp[st2] + lt[st2*hmm.NState+st1] + yp
			}
		}

		// This shift does not change the result due to scale invariance
		mx := floats.Max(terms)
		llf += mx
		floats.AddConst(-mx, terms)

		// Get the probabilities by summing over possible histories.
		for st1 := 0; st1 < hmm.NState; st1++ {
			fprob[j1+st1] = 0
			for st2 := 0; st2 < hmm.NState; st2++ {
				fprob[j1+st1] += math.Exp(terms[st1*hmm.NState+st2])
			}
		}
		mx = normalizeMax(fprob[j1:j1+hmm.NState], 1)
		llf += math.Log(mx)
	}

	hmm.llf[p] = llf
}

// BackwardParticle calculates the backward probabilities for one particle.
func (hmm *HMM) BackwardParticle(p int, wg *sync.WaitGroup) {

	defer wg.Done()

	bprob := hmm.Bprob[p]
	obs := hmm.Obs[p]

	lt := make([]float64, hmm.NState*hmm.NState)
	lby := make([]float64, hmm.NState)
	terms := make([]float64, hmm.NState*hmm.NState)

	// Precompute this
	for j := 0; j < hmm.NState*hmm.NState; j++ {
		lt[j] = math.Log(hmm.Trans[j])
	}

	// Backward sweep
	j0 := hmm.NTime * hmm.NState
	j1 := hmm.NTime*hmm.NState + hmm.NState
	t := hmm.NTime
	for _i := 0; _i < hmm.NTime; _i++ {

		j0 -= hmm.NState
		j1 -= hmm.NState
		t -= 1

		if obs[j0] == NullObs {
			continue
		}

		// Initialize
		if t == hmm.NTime-1 || obs[j1] == NullObs {
			for st := 0; st < hmm.NState; st++ {
				bprob[j0+st] = 1
			}
			continue
		}

		for st := 0; st < hmm.NState; st++ {
			lby[st] = hmm.GetLogObsProb(p, t+1, st) + math.Log(bprob[j1+st])
		}

		// From st1 at t to st2 at t+1.
		for st1 := 0; st1 < hmm.NState; st1++ {
			for st2 := 0; st2 < hmm.NState; st2++ {
				terms[st1*hmm.NState+st2] = lby[st2] + lt[st1*hmm.NState+st2]
			}
		}

		floats.AddConst(-floats.Max(terms), terms)

		// Get the probabilities by summing over possible histories.
		for st1 := 0; st1 < hmm.NState; st1++ {
			bprob[j0+st1] = 0
			for st2 := 0; st2 < hmm.NState; st2++ {
				bprob[j0+st1] += math.Exp(terms[st1*hmm.NState+st2])
			}
		}

		normalizeMax(bprob[j0:j0+hmm.NState], 1)
	}
}

func (hmm *HMM) updateTransParticle(p int, newtrans, logtrans []float64, wg *sync.WaitGroup, mut *sync.Mutex) {

	defer wg.Done()
	obs := hmm.Obs[p]
	fprob := hmm.Fprob[p]
	bprob := hmm.Bprob[p]
	joint := make([]float64, hmm.NState*hmm.NState)
	jointsum := make([]float64, hmm.NState*hmm.NState)
	lcp := make([]float64, hmm.NState)

	for t := 0; t < hmm.NTime-1; t++ {

		if obs[t*hmm.NState] == NullObs || obs[(t+1)*hmm.NState] == NullObs {
			continue
		}

		for st := 0; st < hmm.NState; st++ {
			lcp[st] = hmm.GetLogObsProb(p, t+1, st) + math.Log(bprob[t*hmm.NState+st])
		}

		// Get the joint probabilities on the log scale
		for st1 := 0; st1 < hmm.NState; st1++ {
			lfp := math.Log(fprob[t*hmm.NState+st1])
			for st2 := 0; st2 < hmm.NState; st2++ {
				joint[st1*hmm.NState+st2] = lfp + lcp[st2] + logtrans[st1*hmm.NState+st2]
			}
		}

		// Convert to probabilities
		floats.AddConst(-floats.Max(joint), joint)
		for j := range joint {
			joint[j] = math.Exp(joint[j])
		}
		normalizeSum(joint, 0)

		floats.Add(jointsum, joint)
	}

	mut.Lock()
	floats.Add(newtrans, jointsum)
	mut.Unlock()
}

// UpdateTrans updates the transmission probability matrix.
func (hmm *HMM) UpdateTrans() {

	newtrans := make([]float64, hmm.NState*hmm.NState)

	logtrans := make([]float64, hmm.NState*hmm.NState)
	for j := range hmm.Trans {
		logtrans[j] = math.Log(hmm.Trans[j])
	}

	var wg sync.WaitGroup
	var mut sync.Mutex
	for p := 0; p < hmm.NParticle; p++ {
		wg.Add(1)
		go hmm.updateTransParticle(p, newtrans, logtrans, &wg, &mut)
	}
	wg.Wait()

	// Normalize to probabilties by row
	for st := 0; st < hmm.NState; st++ {
		normalizeSum(newtrans[st*hmm.NState:(st+1)*hmm.NState], 1/float64(hmm.NState))
	}

	hmm.Trans = newtrans
}

// OracleTrans estimates the transmission probabilities from a known sequence of
// state values.
func (hmm *HMM) OracleTrans() []float64 {

	tr := make([]float64, hmm.NState*hmm.NState)

	for p := 0; p < hmm.NParticle; p++ {
		obs := hmm.Obs[p]
		for t := 0; t < hmm.NTime-1; t++ {

			if obs[t*hmm.NState] == NullObs || obs[(t+1)*hmm.NState] == NullObs {
				continue
			}

			st1 := hmm.State[p][t]
			st2 := hmm.State[p][t+1]

			tr[st1*hmm.NState+st2]++
		}
	}

	for st := 0; st < hmm.NState; st++ {
		normalizeSum(tr[st*hmm.NState:(st+1)*hmm.NState], 1/float64(hmm.NState))
	}

	return tr
}

// OracleZprob estimates the probabilities that the observed value is zero using
// a known sequence of state values.
func (hmm *HMM) OracleZprob() []float64 {

	zpr := make([]float64, hmm.NState*hmm.NState)
	den := make([]float64, hmm.NState)

	for p := 0; p < hmm.NParticle; p++ {
		obs := hmm.Obs[p]
		for t := 0; t < hmm.NTime-1; t++ {

			if obs[t*hmm.NState] == NullObs {
				continue
			}

			st1 := hmm.State[p][t]
			den[st1] += 1
			for st2 := 0; st2 < hmm.NState; st2++ {
				if obs[t*hmm.NState+st2] == 0 {
					zpr[st1*hmm.NState+st2] += 1
				}
			}
		}
	}

	for st1 := 0; st1 < hmm.NState; st1++ {
		i := st1 * hmm.NState
		floats.Scale(1/den[st1], zpr[i:i+hmm.NState])
	}

	return zpr
}

// MarginalMoments calculates the mean and standard deviation
// for each observed component, given that the value is not equal
// to zero.
func (hmm *HMM) MarginalMoments() ([]float64, []float64) {

	mean := make([]float64, hmm.NState)
	num := make([]float64, hmm.NState)
	for p := 0; p < hmm.NParticle; p++ {
		i := 0
		obs := hmm.Obs[p]
		for t := 0; t < hmm.NTime; t++ {
			if obs[i] == NullObs {
				i += hmm.NState
				continue
			}
			for st := 0; st < hmm.NState; st++ {
				if obs[i] != 0 {
					mean[st] += obs[i]
					num[st]++
				}
				i++
			}
		}
	}
	floats.Div(mean, num)

	std := make([]float64, hmm.NState)
	for p := 0; p < hmm.NParticle; p++ {
		i := 0
		obs := hmm.Obs[p]
		for t := 0; t < hmm.NTime; t++ {
			if obs[t*hmm.NState] == NullObs {
				i += hmm.NState
				continue
			}
			for st := 0; st < hmm.NState; st++ {
				y := obs[t*hmm.NState+st]
				if !hmm.ZeroInflated || y != 0 {
					y -= mean[st]
					std[st] += y * y
				}
				i++
			}
		}
	}
	floats.Div(std, num)

	for i := range std {
		if math.IsNaN(mean[i]) || math.IsNaN(std[i]) {
			std[i] = 1
			mean[i] = 0
		}
		std[i] = math.Sqrt(std[i])
	}

	return mean, std
}

// OracleMoments estimates the means and standard deviations
// for non-zero emissions using known state values.
func (hmm *HMM) OracleMoments() ([]float64, []float64) {

	mean := make([]float64, hmm.NState*hmm.NState)
	std := make([]float64, hmm.NState*hmm.NState)
	den := make([]float64, hmm.NState*hmm.NState)

	for p := 0; p < hmm.NParticle; p++ {
		obs := hmm.Obs[p]
		for t := 0; t < hmm.NTime; t++ {
			if obs[t*hmm.NState] == NullObs {
				continue
			}
			st1 := hmm.State[p][t] // true state
			for st2 := 0; st2 < hmm.NState; st2++ {
				y := obs[t*hmm.NState+st2]
				if !hmm.ZeroInflated || y != 0 {
					mean[st1*hmm.NState+st2] += y
					std[st1*hmm.NState+st2] += y * y
					den[st1*hmm.NState+st2] += 1
				}
			}
		}
	}

	for st1 := 0; st1 < hmm.NState; st1++ {
		j := st1 * hmm.NState
		for st2 := 0; st2 < hmm.NState; st2++ {
			mean[j] /= den[j]
			std[j] /= den[j]
			j += 1
		}
	}

	for st1 := 0; st1 < hmm.NState; st1++ {
		i := st1 * hmm.NState
		for st2 := 0; st2 < hmm.NState; st2++ {
			m := mean[i]
			std[i] -= m * m
			std[i] = math.Sqrt(std[i])
			i += 1
		}
	}

	return mean, std
}

func (hmm *HMM) WriteOracleSummary(labels []string) {

	hmm.parlogger.Printf("\nOracle statistics:\n")

	hmm.parlogger.Printf("Initial state distribution:\n")
	ist := hmm.OracleInit()
	hmm.writeMatrix(ist, 0, hmm.NState, 1, labels)
	hmm.parlogger.Printf("\n")

	hmm.parlogger.Printf("Transition matrix:\n")
	tr := hmm.OracleTrans()
	hmm.writeMatrix(tr, 0, hmm.NState, hmm.NState, labels)
	hmm.parlogger.Printf("\n")

	if hmm.ZeroInflated {
		hmm.parlogger.Printf("Zero probabilities:\n")
		zpr := hmm.OracleZprob()
		hmm.writeMatrix(zpr, 0, hmm.NState, hmm.NState, labels)
		hmm.parlogger.Printf("\n")
	}

	hmm.parlogger.Printf("Means:\n")
	mean, sd := hmm.OracleMoments()
	hmm.writeMatrix(mean, 0, hmm.NState, hmm.NState, labels)
	hmm.parlogger.Printf("\n")

	hmm.parlogger.Printf("Standard deviations:\n")
	hmm.writeMatrix(sd, 0, hmm.NState, hmm.NState, labels)
	hmm.parlogger.Printf("\n")
}

// ObservedInit returns the frequency distribution of initial states.
func (hmm *HMM) OracleInit() []float64 {

	v := make([]float64, hmm.NState)

	for _, u := range hmm.State {
		// Use first non-null value for each particle
		for j := range u {
			if u[j] != NullState {
				v[u[j]]++
				break
			}
		}
	}

	normalizeSum(v, 1/float64(hmm.NState))
	return v
}

func (hmm *HMM) UpdateInit() {

	vt := make([]float64, hmm.NState)
	zero(hmm.Init)

	for p := 0; p < hmm.NParticle; p++ {
		for t := 0; t < hmm.NTime; t++ {

			if hmm.Obs[p][t*hmm.NState] != NullObs {
				fprob := hmm.Fprob[p][t*hmm.NState : (t+1)*hmm.NState]
				bprob := hmm.Bprob[p][t*hmm.NState : (t+1)*hmm.NState]

				floats.MulTo(vt, fprob, bprob)
				normalizeSum(vt, 0)
				floats.Add(hmm.Init, vt)
			}
		}
	}

	normalizeSum(hmm.Init, 1/float64(hmm.NState))
}

func (hmm *HMM) UpdateObsParams() {

	switch hmm.ObsModelForm {
	case Gaussian:
		hmm.updateGaussianObsParams()
	case Poisson:
		hmm.updatePoissonObsParams()
	case Tweedie:
		// Can use the same update as Poisson
		hmm.updatePoissonObsParams()
	default:
		panic("Unkown observation model")
	}
}

func (hmm *HMM) updateGaussianObsParams() {

	pr := make([]float64, hmm.NState)
	pt := make([]float64, hmm.NState)

	zero(hmm.Mean)
	zero(hmm.Zprob)

	for p := 0; p < hmm.NParticle; p++ {
		i := 0
		obs := hmm.Obs[p]
		for t := 0; t < hmm.NTime; t++ {

			floats.MulTo(pr, hmm.Fprob[p][i:i+hmm.NState], hmm.Bprob[p][i:i+hmm.NState])
			normalizeSum(pr, 0)
			floats.Add(pt, pr)

			if obs[t*hmm.NState] != NullObs {
				for st1 := 0; st1 < hmm.NState; st1++ {
					for st2 := 0; st2 < hmm.NState; st2++ {
						y := obs[t*hmm.NState+st2]
						if hmm.ZeroInflated {
							if y == 0.0 {
								hmm.Zprob[st1*hmm.NState+st2] += pr[st1]
							} else {
								hmm.Mean[st1*hmm.NState+st2] += pr[st1] * y
							}
						} else {
							hmm.Mean[st1*hmm.NState+st2] += pr[st1] * y
						}
					}
				}
			}

			i += hmm.NState
		}
	}

	for st := 0; st < hmm.NState; st++ {
		i, j := st*hmm.NState, (st+1)*hmm.NState
		var s = 1 / pt[st]
		if pt[st] < 1e-10 {
			hmm.msglogger.Printf("Underflow in Gaussian SD update")
			s = 0
		}
		if hmm.ZeroInflated {
			floats.Scale(s, hmm.Zprob[i:j])
		}
		floats.Scale(s, hmm.Mean[i:j])
	}

	if hmm.ZeroInflated {
		// Truncate the zero-probabilities
		for i := range hmm.Zprob {
			if hmm.Zprob[i] > zpmax {
				hmm.Zprob[i] = zpmax
			}
		}

		for i := 0; i < hmm.NState*hmm.NState; i++ {
			hmm.Mean[i] /= (1.0 - hmm.Zprob[i])
		}
	}
}

func (hmm *HMM) updatePoissonObsParams() {

	pr := make([]float64, hmm.NState)
	pt := make([]float64, hmm.NState)

	zero(hmm.Mean)

	for p := 0; p < hmm.NParticle; p++ {
		i := -hmm.NState
		obs := hmm.Obs[p]
		for t := 0; t < hmm.NTime; t++ {

			i += hmm.NState

			if obs[i] == NullObs {
				continue
			}

			floats.MulTo(pr, hmm.Fprob[p][i:i+hmm.NState], hmm.Bprob[p][i:i+hmm.NState])
			normalizeSum(pr, 0)
			floats.Add(pt, pr)
			for st1 := 0; st1 < hmm.NState; st1++ {
				for st2 := 0; st2 < hmm.NState; st2++ {
					y := obs[t*hmm.NState+st2]
					hmm.Mean[st1*hmm.NState+st2] += pr[st1] * y
				}
			}
		}
	}

	for st := 0; st < hmm.NState; st++ {
		i, j := st*hmm.NState, (st+1)*hmm.NState
		var s = 1 / pt[st]
		if pt[st] < 1e-10 {
			hmm.msglogger.Printf("Underflow in Poisson update...")
			s = 0
		}
		floats.Scale(s, hmm.Mean[i:j])
	}
}

func (hmm *HMM) UpdateStdFree() {

	pr := make([]float64, hmm.NState)
	pt := make([]float64, hmm.NState)

	zero(hmm.Std)

	for p := 0; p < hmm.NParticle; p++ {
		i := 0
		obs := hmm.Obs[p]
		for t := 0; t < hmm.NTime; t++ {

			floats.MulTo(pr, hmm.Fprob[p][i:i+hmm.NState], hmm.Bprob[p][i:i+hmm.NState])
			normalizeSum(pr, 0)
			floats.Add(pt, pr)

			if obs[t*hmm.NState] != NullObs {
				for st1 := 0; st1 < hmm.NState; st1++ {
					for st2 := 0; st2 < hmm.NState; st2++ {
						y := obs[t*hmm.NState+st2]
						if y != 0.0 {
							y -= hmm.Mean[st1*hmm.NState+st2]
							hmm.Std[st1*hmm.NState+st2] += pr[st1] * y * y
						}
					}
				}
			}

			i += hmm.NState
		}
	}

	for st := 0; st < hmm.NState; st++ {
		i, j := st*hmm.NState, (st+1)*hmm.NState
		var s = 1 / pt[st]
		if pt[st] < 1e-10 {
			print("Underflow in Std update")
			s = 0
		}
		floats.Scale(s, hmm.Std[i:j])
	}

	for i := 0; i < hmm.NState*hmm.NState; i++ {
		if hmm.ZeroInflated {
			hmm.Std[i] = math.Sqrt(hmm.Std[i] / (1.0 - hmm.Zprob[i]))
		} else {
			hmm.Std[i] = math.Sqrt(hmm.Std[i])
		}
		if hmm.Std[i] < sdmin {
			hmm.Std[i] = sdmin
		}
	}
}

func (hmm *HMM) UpdateStdConst() {

	pr := make([]float64, hmm.NState)

	zero(hmm.Std)

	pt := 0
	for p := 0; p < hmm.NParticle; p++ {
		i := 0
		obs := hmm.Obs[p]
		for t := 0; t < hmm.NTime; t++ {

			if obs[t*hmm.NState] != NullObs {

				floats.MulTo(pr, hmm.Fprob[p][i:i+hmm.NState], hmm.Bprob[p][i:i+hmm.NState])
				normalizeSum(pr, 0)

				for st1 := 0; st1 < hmm.NState; st1++ {
					for st2 := 0; st2 < hmm.NState; st2++ {
						y := hmm.Obs[p][t*hmm.NState+st2]
						if y != 0.0 {
							y -= hmm.Mean[st1*hmm.NState+st2]
							hmm.Std[st2] += pr[st1] * y * y
						}
					}
				}
				pt++
			}

			i += hmm.NState
		}
	}

	floats.Scale(1/float64(pt), hmm.Std[0:hmm.NState])

	if hmm.ZeroInflated {
		for i := range hmm.Std {
			hmm.Std[i] = math.Sqrt(hmm.Std[i] / (1.0 - hmm.Zprob[i]))
		}
	} else {
		for i := range hmm.Std {
			hmm.Std[i] = math.Sqrt(hmm.Std[i])
		}
	}

	// Cap the SD parameters
	for i := range hmm.Std {
		if hmm.Std[i] < sdmin {
			hmm.Std[i] = sdmin
		}
	}

	// The remaining rows are constrained to be equal to the first row.
	for i := 1; i < hmm.NState; i++ {
		j := i * hmm.NState
		copy(hmm.Std[j:j+hmm.NState], hmm.Std[0:hmm.NState])
	}
}

// Fit uses the EM algorithm to estimate the structural parameters of the HMM.
func (hmm *HMM) Fit(maxiter int) {

	hmm.LLF = make([]float64, 0, maxiter)

	hmm.msglogger.Printf("Estimating model parameters...\n")
	bar := progressbar.New(4 * maxiter)
	var llf float64

	for i := 0; i < maxiter; i++ {
		_ = bar.Add(1)
		hmm.msglogger.Printf("Beginning ForwardBackward...")
		hmm.ForwardBackward()
		_ = bar.Add(1)
		hmm.msglogger.Printf("Beginning UpdateTrans...")
		hmm.UpdateTrans()
		_ = bar.Add(1)
		hmm.msglogger.Printf("Beginning UpdateInit...")
		hmm.UpdateInit()
		_ = bar.Add(1)
		hmm.msglogger.Printf("Beginning UpdateObsParams...")
		hmm.UpdateObsParams()

		if hmm.ObsModelForm == Gaussian {
			hmm.msglogger.Printf("Updating variance parameters...")
			switch hmm.VarForm {
			case VarFree:
				hmm.UpdateStdFree()
			case VarConst:
				hmm.UpdateStdConst()
			default:
				panic("not implemented")
			}
		}

		llfnew := floats.Sum(hmm.llf)
		if i > 0 {
			if llfnew < llf {
				hmm.msglogger.Printf("Log-likelihood increased by %f\n", llf-llfnew)
			} else if llfnew-llf < 1e-8 {
				// converged
				break
			}
		}

		llf = llfnew
		hmm.msglogger.Printf("llf=%f\n", llf)
	}

	hmm.LLF = append(hmm.LLF, llf)
}

// WriteSummary writes the model parameters to the given writer.
// The optional row labels are used if provided.
func (hmm *HMM) WriteSummary(labels []string, title string) {

	hmm.parlogger.Printf(title)
	hmm.parlogger.Printf("\n")

	hmm.parlogger.Printf("Initial states distribution:\n")
	hmm.writeMatrix(hmm.Init, 0, hmm.NState, 1, labels)
	hmm.parlogger.Printf("\n")

	hmm.parlogger.Printf("Transition matrix:\n")
	hmm.writeMatrix(hmm.Trans, 0, hmm.NState, hmm.NState, labels)
	hmm.parlogger.Printf("\n")

	if hmm.ZeroInflated {
		hmm.parlogger.Printf("Zero probabilties:\n")
		hmm.writeMatrix(hmm.Zprob, 0, hmm.NState, hmm.NState, labels)
		hmm.parlogger.Printf("\n")
	}

	hmm.parlogger.Printf("Means:\n")
	hmm.writeMatrix(hmm.Mean, 0, hmm.NState, hmm.NState, labels)
	hmm.parlogger.Printf("\n")

	if hmm.ObsModelForm == Gaussian {
		hmm.parlogger.Printf("Standard deviations:\n")
		hmm.writeMatrix(hmm.Std, 0, hmm.NState, hmm.NState, labels)
		hmm.parlogger.Printf("\n")
	}
}

// writeMatrix writes a matrix in text format to the logger
func (hmm *HMM) writeMatrix(x []float64, off, nrow, ncol int, labels []string) {

	var buf bytes.Buffer

	for i := 0; i < nrow; i++ {

		buf.Reset()

		if labels != nil {
			_, _ = io.WriteString(&buf, fmt.Sprintf("%-20s", labels[i]))
		}
		for j := 0; j < ncol; j++ {
			_, _ = io.WriteString(&buf, fmt.Sprintf("%12.4f ", x[off+i*ncol+j]))
		}

		hmm.parlogger.Printf(buf.String())
	}
}

// normalize the values in x from index i to index i+q to have a maxiumum of 1.
func normalizeMax(x []float64, z float64) float64 {
	scale := floats.Max(x)
	if scale < 1e-10 {
		for j := range x {
			x[j] = z
		}
		return 0
	}
	floats.Scale(1/scale, x)
	return scale
}

// Subtract the maximum value from x, then exponentiate.
func normalizeMaxLog(x []float64) float64 {
	mx := floats.Max(x)
	floats.AddConst(-mx, x)
	for j := range x {
		x[j] = math.Exp(x[j])
	}

	return mx
}

// normalize the values in x from index i to index i+q to have a sum of 1.
func normalizeSum(x []float64, z float64) {
	scale := floats.Sum(x)
	if scale < 1e-10 {
		for j := range x {
			x[j] = z
		}
		return
	}
	floats.Scale(1/scale, x)
}

func argmax(x []float64) int {
	j := 0
	v := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > v {
			v = x[i]
			j = i
		}
	}

	return j
}

// Zero the elements of x
func zero(x []float64) {
	for j := range x {
		x[j] = 0
	}
}

// Generate a discrete random variable from the given probability vector,
// which must sum to 1.
func genDiscrete(pr []float64) int {

	u := rand.Float64()
	p := 0.0
	for j := range pr {
		p += pr[j]
		if u < p {
			return j
		}
	}

	// Can't reach here
	panic("Not a probability vector")
}

// makeIntArray makes a collection of r slices
// of length c, packed contiguously.
func makeIntArray(r, c int) [][]int {

	bka := make([]int, r*c)
	x := make([][]int, r)
	ii := 0
	for j := 0; j < r; j++ {
		x[j] = bka[ii : ii+c]
		ii += c
	}

	return x
}

// makeFloatArray makes a collection of r slices
// of length c, packed contiguously.
func makeFloatArray(r, c int) [][]float64 {

	bka := make([]float64, r*c)
	x := make([][]float64, r)
	ii := 0
	for j := 0; j < r; j++ {
		x[j] = bka[ii : ii+c]
		ii += c
	}

	return x
}

func genPoisson(lambda float64) float64 {

	if lambda <= 0 {
		panic("lambda <= 0 in Poisson")
	}

	L := math.Exp(-lambda)
	var k int64 = 0
	var p float64 = 1.0

	for p > L {
		k++
		p *= rand.Float64()
	}

	return float64(k - 1)
}

// Generate a Gamma random variable with mean alp*bet and variance alp*bet^2.
func genGamma(alp, bet float64) float64 {

	d := alp - 1/3.
	c := 1 / math.Sqrt(9*d)

	for {
		z := rand.NormFloat64()
		u := rand.Float64()
		v := math.Pow(1+c*z, 3)
		if z > -1/c && math.Log(u) < z*z/2+d-d*v+d*math.Log(v) {
			return bet * d * v
		}
	}
}

// Generate a Tweedie random variable with mean mu and variance sig2 * mu^p.
func genTweedie(mu, sig2, p float64) float64 {

	if p <= 1 || p >= 2 {
		panic("p must be between 1 and 2")
	}

	lam := math.Pow(mu, 2-p) / ((2 - p) * sig2)
	alp := (2 - p) / (p - 1)
	bet := math.Pow(mu, 1-p) / ((p - 1) * sig2)

	n := genPoisson(lam)

	var z float64
	for k := 0; k < int(n); k++ {
		z += genGamma(alp, 1/bet)
	}

	return z
}

// ReadHMM reads a compressed MultiHMM value from a gzip-compressed
// gob file.
func ReadHMM(fname string) *MultiHMM {

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

	var hmm MultiHMM
	if err := dec.Decode(&hmm); err != nil {
		panic(err)
	}

	return &hmm
}

// CompareStates returns the number of positions where the state
// sequences x and y agree, and the number of positions in which
// x and y are not null.  Panics if the lengths of x and y differ,
// or if there is any position in which exactly one of x and y is
// null.
func CompareStates(x, y []int) (int, int) {

	if len(x) != len(y) {
		panic("Lengths are not equal")
	}

	var e, n int
	for t := range x {
		if x[t] == NullState && y[t] != NullState {
			panic("inconsistent")
		}
		if x[t] != NullState && y[t] == NullState {
			panic("inconsistent")
		}
		if x[t] == NullState {
			continue
		}
		if x[t] != y[t] {
			e++
		}
		n++
	}

	return e, n
}
