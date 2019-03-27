package hmmlib

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"sync"

	"github.com/schollz/progressbar"
	"gonum.org/v1/gonum/floats"
)

const (
	// Maximum allowed value for Zprob
	zpmax float64 = 0.95

	// Minimum allowed value for the observation SD
	sdmin = 1e-8

	// Null value for an observation.  When the observed value is Null, it is
	// not used in estimation.  If the first component of the observed vector
	// is Null, the are all treated as Null.
	NullObs float64 = -9999

	// Null value for a state
	NullState int = -9999
)

// VarType indicates how the variance structure is modeled.
type VarType uint8

const (
	VarFree VarType = iota
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
	VarForm VarType

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

	// The states
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
	LLF float64

	// The log-likelihood function for one particle
	llf []float64

	// Additional parameters for the observation model
	Aux []float64
}

// New returns an HMM value with the given size parameters.
func New(NParticle, NState, NTime int) *HMM {

	return &HMM{
		NParticle: NParticle,
		NTime:     NTime,
		NState:    NState,
	}
}

// Initialize allocates workspaces for parameter estimation.  Call
// this prior to calling Fit.
func (hmm *HMM) Initialize() {
	hmm.Bprob = makeFloatArray(hmm.NParticle, hmm.NState*hmm.NTime)
	hmm.Fprob = makeFloatArray(hmm.NParticle, hmm.NState*hmm.NTime)
	hmm.llf = make([]float64, hmm.NParticle)
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
		if t == 0 || (obs[j1] != NullObs && obs[j0] == NullObs) {
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
		for i := 0; i < hmm.NState; i++ {
			for j := 0; j < hmm.NState; j++ {
				if i == j {
					hmm.Std[i*hmm.NState+j] = std[i]
				} else {
					hmm.Std[i*hmm.NState+j] = std[i] / 5
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
					hmm.Zprob[i*hmm.NState+j] = 0.7
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
		return math.Inf(-1)
	}

	pw := hmm.Aux[0]

	var lpr float64

	ii := st * hmm.NState
	for st1 := 0; st1 < hmm.NState; st1++ {
		y := obs[t*hmm.NState+st1]
		mn := hmm.Mean[ii]
		lmn := math.Log(mn)
		lpr += y*math.Exp((1-pw)*lmn)/(1-pw) - math.Exp((2-pw)*lmn)/(2-pw)
		ii++
	}

	return lpr
}

func (hmm *HMM) getPoissonLogObsProb(p, t, st int) float64 {

	obs := hmm.Obs[p]

	if obs[t*hmm.NState] == NullObs {
		return math.Inf(-1)
	}

	var lpr float64
	ii := st * hmm.NState
	for st1 := 0; st1 < hmm.NState; st1++ {
		y := obs[t*hmm.NState+st1]
		mn := hmm.Mean[ii]
		lpr += -mn + y*math.Log(mn)
		ii++
	}

	return lpr
}

func (hmm *HMM) getGaussianLogObsProb(p, t, st int) float64 {

	obs := hmm.Obs[p]

	if obs[t*hmm.NState] == NullObs {
		return math.Inf(-1)
	}

	var lpr float64
	ii := st * hmm.NState
	for st1 := 0; st1 < hmm.NState; st1++ {
		y := obs[t*hmm.NState+st1]
		zpr := hmm.Zprob[ii]
		mn := hmm.Mean[ii]
		sd := hmm.Std[ii]
		if y == 0 {
			lpr += math.Log(zpr)
		} else {
			z := (y - mn) / sd
			q := -math.Log(sd) - z*z/2
			lpr += q + math.Log(1.0-zpr)
		}
		ii++
	}

	return lpr
}

// GenStates generates a random state sequence.
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
						u := rand.NormFloat64()
						hmm.Obs[p][t*hmm.NState+j] = hmm.Mean[ip+j] + u*hmm.Std[ip+j]
					case Poisson:
						hmm.Obs[p][t*hmm.NState+j] = genPoisson(hmm.Mean[ip+j])
					case Tweedie:
						// Use Poisson for now
						hmm.Obs[p][t*hmm.NState+j] = genPoisson(hmm.Mean[ip+j])
					default:
						panic("unkown model\n")
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
		if t == 0 || (obs[j0] == NullObs && obs[j1] != NullObs) {
			for st := 0; st < hmm.NState; st++ {
				fprob[st] = math.Log(hmm.Init[st]) + hmm.GetLogObsProb(p, t, st)
			}
			llf += normalizeMaxLog(fprob[j1 : j1+hmm.NState])
			continue
		}

		// Precompute this
		for st := 0; st < hmm.NState; st++ {
			lfp[st] = math.Log(fprob[j0+st])
		}

		// Calculate components of the update on the log scale.
		// Transition is from From st2 at t-1 to st1 at t.
		for st1 := 0; st1 < hmm.NState; st1++ {
			yp := hmm.GetLogObsProb(p, t, st1)
			for st2 := 0; st2 < hmm.NState; st2++ {
				terms[st1*hmm.NState+st2] = lfp[st2] + lt[st2*hmm.NState+st1] + yp
			}
		}

		// Does not change result due to scale invariance
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

// BackwardParticle calculates the backward probabilites for one particle.
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
		if t == hmm.NTime-1 || (obs[j1] == NullObs && obs[j0] != NullObs) {
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

// UpdateTrans updates the transmission probability matrix.
func (hmm *HMM) UpdateTrans() {

	newtrans := make([]float64, hmm.NState*hmm.NState)
	rtrans := make([]float64, hmm.NState*hmm.NState)
	lcp := make([]float64, hmm.NState)

	for p := 0; p < hmm.NParticle; p++ {
		obs := hmm.Obs[p]
		for t := 0; t < hmm.NTime-1; t++ {

			if obs[t*hmm.NState] == NullObs || obs[(t+1)*hmm.NState] == NullObs {
				continue
			}

			for st := 0; st < hmm.NState; st++ {
				lcp[st] = hmm.GetLogObsProb(p, t+1, st)
			}

			for st1 := 0; st1 < hmm.NState; st1++ {
				u := math.Log(hmm.Fprob[p][t*hmm.NState+st1])
				for st2 := 0; st2 < hmm.NState; st2++ {
					z := u + math.Log(hmm.Bprob[p][t*hmm.NState+st2])
					z += math.Log(hmm.Trans[st1*hmm.NState+st2])
					z += lcp[st2]
					rtrans[st1*hmm.NState+st2] = z
				}
			}

			floats.AddConst(-floats.Max(rtrans), rtrans)
			for j := range rtrans {
				rtrans[j] = math.Exp(rtrans[j])
			}

			normalizeSum(rtrans, 0)
			floats.Add(newtrans, rtrans)
		}
	}

	// Normalize to probabilties by row
	for st := 0; st < hmm.NState; st++ {
		normalizeSum(newtrans[st*hmm.NState:(st+1)*hmm.NState], 1/float64(hmm.NState))
	}

	hmm.Trans = newtrans
}

// OracleTrans estimates the transmission probabilites from a known sequence of
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
	nn := make([]float64, hmm.NState)
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
					nn[st]++
				}
				i++
			}
		}
	}
	floats.Div(mean, nn)

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
	floats.Div(std, nn)

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
					if hmm.ObsModelForm == Gaussian {
						std[st1*hmm.NState+st2] += y * y
					}
					den[st1*hmm.NState+st2] += 1
				}
			}
		}
	}

	for st1 := 0; st1 < hmm.NState; st1++ {
		j := st1 * hmm.NState
		for st2 := 0; st2 < hmm.NState; st2++ {
			mean[j] /= den[j]
			if hmm.ObsModelForm == Gaussian {
				std[j] /= den[j]
			}
			j += 1
		}
	}

	if hmm.ObsModelForm == Gaussian {
		for st1 := 0; st1 < hmm.NState; st1++ {
			i := st1 * hmm.NState
			for st2 := 0; st2 < hmm.NState; st2++ {
				m := mean[i]
				std[i] -= m * m
				std[i] = math.Sqrt(std[i])
				i += 1
			}
		}
	}

	return mean, std
}

func minmax(x []int) (int, int) {
	mn, mx := x[0], x[0]
	for j := 1; j < len(x); j++ {
		if x[j] < mn {
			mn = x[j]
		}
		if x[j] > mx {
			mx = x[j]
		}
	}
	return mn, mx
}

func (hmm *HMM) WriteOracleSummary(labels []string, w io.Writer) {

	io.WriteString(w, "\nOracle statistics:\n")

	io.WriteString(w, fmt.Sprintf("%d particles\n", hmm.NParticle))
	io.WriteString(w, fmt.Sprintf("%d time points per particle\n", hmm.NTime))
	io.WriteString(w, fmt.Sprintf("%d states\n", hmm.NState))

	io.WriteString(w, "Initial state distribution:\n")
	ist := hmm.OracleInit()
	writeMatrix(ist, 0, hmm.NState, 1, labels, w)
	io.WriteString(w, "\n")

	io.WriteString(w, "Transition matrix:\n")
	tr := hmm.OracleTrans()
	writeMatrix(tr, 0, hmm.NState, hmm.NState, labels, w)
	io.WriteString(w, "\n")

	if hmm.ZeroInflated {
		io.WriteString(w, "Zero probabilities:\n")
		zpr := hmm.OracleZprob()
		writeMatrix(zpr, 0, hmm.NState, hmm.NState, labels, w)
		io.WriteString(w, "\n")
	}

	io.WriteString(w, "Means:\n")
	mean, sd := hmm.OracleMoments()
	writeMatrix(mean, 0, hmm.NState, hmm.NState, labels, w)
	io.WriteString(w, "\n")

	if hmm.ObsModelForm == Gaussian {
		io.WriteString(w, "Standard deviations:\n")
		writeMatrix(sd, 0, hmm.NState, hmm.NState, labels, w)
		io.WriteString(w, "\n")
	}
}

// ObservedInit returns the frequency distribution of initial states.
func (hmm *HMM) OracleInit() []float64 {

	v := make([]float64, hmm.NState)

	for _, u := range hmm.State {
		if u[0] != NullState {
			v[u[0]]++
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
						if y == 0.0 {
							hmm.Zprob[st1*hmm.NState+st2] += pr[st1]
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
			print("Underflow in Std update")
			s = 0
		}
		floats.Scale(s, hmm.Zprob[i:j])
		floats.Scale(s, hmm.Mean[i:j])
	}

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

func (hmm *HMM) updatePoissonObsParams() {

	pr := make([]float64, hmm.NState)
	pt := make([]float64, hmm.NState)

	zero(hmm.Mean)

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
						hmm.Mean[st1*hmm.NState+st2] += pr[st1] * y
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
			print("Underflow in Poisson update...")
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
		hmm.Std[i] = math.Sqrt(hmm.Std[i] / (1.0 - hmm.Zprob[i]))
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

	for i := 0; i < hmm.NState*hmm.NState; i++ {
		hmm.Std[i] = math.Sqrt(hmm.Std[i] / (1.0 - hmm.Zprob[i]))
		if hmm.Std[i] < sdmin {
			hmm.Std[i] = sdmin
		}
	}

	// The remaining roles are constrained to be equal to the first row.
	for i := 1; i < hmm.NState; i++ {
		j := i * hmm.NState
		copy(hmm.Std[j:j+hmm.NState], hmm.Std[0:hmm.NState])
	}
}

// Fit uses the EM algorithm to estimate the structural parameters of the HMM.
func (hmm *HMM) Fit(iter int) {

	print("Estimating model parameters...\n")
	bar := progressbar.New(iter)
	var llf float64

	for i := 0; i < iter; i++ {
		bar.Add(1)
		hmm.ForwardBackward()
		hmm.UpdateTrans()
		hmm.UpdateInit()
		hmm.UpdateObsParams()

		if hmm.ObsModelForm == Gaussian {
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
				print("Log-likelihood decreased\n")
			} else if llfnew-llf < 1e-8 {
				// converged
				break
			}
		}

		llf = llfnew
	}

	hmm.LLF = llf
}

// WriteSummary writes the model parameters to the given writer.
// The optional row labels are used if provided.
func (hmm *HMM) WriteSummary(labels []string, title string, w io.Writer) {

	io.WriteString(w, title)
	io.WriteString(w, "\n\n")

	io.WriteString(w, "Initial states distribution:\n")
	writeMatrix(hmm.Init, 0, hmm.NState, 1, labels, w)
	io.WriteString(w, "\n")

	io.WriteString(w, "Transition matrix:\n")
	writeMatrix(hmm.Trans, 0, hmm.NState, hmm.NState, labels, w)
	io.WriteString(w, "\n")

	if hmm.ZeroInflated {
		io.WriteString(w, "Zero probabilties:\n")
		writeMatrix(hmm.Zprob, 0, hmm.NState, hmm.NState, labels, w)
		io.WriteString(w, "\n")
	}

	io.WriteString(w, "Means:\n")
	writeMatrix(hmm.Mean, 0, hmm.NState, hmm.NState, labels, w)
	io.WriteString(w, "\n")

	if hmm.ObsModelForm == Gaussian {
		io.WriteString(w, "Standard deviations:\n")
		writeMatrix(hmm.Std, 0, hmm.NState, hmm.NState, labels, w)
		io.WriteString(w, "\n")
	}
}

// writeMatrix writes a matrix in text format to the given writer
func writeMatrix(x []float64, off, nrow, ncol int, labels []string, w io.Writer) {

	for i := 0; i < nrow; i++ {
		if labels != nil {
			io.WriteString(w, fmt.Sprintf("%-20s", labels[i]))
		}
		for j := 0; j < ncol; j++ {
			io.WriteString(w, fmt.Sprintf("%12.4f ", x[off+i*ncol+j]))
		}
		io.WriteString(w, "\n")
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

func xxxgenPoisson(lambda float64) float64 {

	c := 0.767 - 3.36/lambda
	beta := math.Pi / math.Sqrt(3.0*lambda)
	alpha := beta * lambda
	k := math.Log(c) - lambda - math.Log(beta)

	for {
		u := rand.Float64()
		x := (alpha - math.Log((1.0-u)/u)) / beta
		n := math.Floor(x + 0.5)
		if n < 0 {
			continue
		}
		v := rand.Float64()
		y := alpha - beta*x
		z := 1.0 + math.Exp(y)
		lhs := y + math.Log(v/(z*z))
		rhs := k + n*math.Log(lambda)
		lg, _ := math.Lgamma(n + 1)
		rhs -= lg
		if lhs <= rhs {
			return n
		}
	}
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
