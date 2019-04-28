package hmmsim

import (
	"math"
	"math/rand"

	"github.com/kshedden/multihmm/hmmlib"
)

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

// GenObs generates a random observation sequence.
func GenObs(hmm *hmmlib.HMM) {

	hmm.Obs = makeFloatArray(hmm.NParticle, hmm.NState*hmm.NTime)

	for p := 0; p < hmm.NParticle; p++ {
		for t := 0; t < hmm.NTime; t++ {

			st := hmm.State[p][t]
			ip := st * hmm.NState

			if st == hmmlib.NullState {
				for j := 0; j < hmm.NState; j++ {
					hmm.Obs[p][t*hmm.NState+j] = hmmlib.NullObs
				}
			} else {
				for j := 0; j < hmm.NState; j++ {
					switch hmm.ObsModelForm {
					case hmmlib.Gaussian:
						if hmm.ZeroInflated && rand.Float64() < hmm.Zprob[ip+j] {
							hmm.Obs[p][t*hmm.NState+j] = 0
						} else {
							u := rand.NormFloat64()
							hmm.Obs[p][t*hmm.NState+j] = hmm.Mean[ip+j] + u*hmm.Std[ip+j]
						}
					case hmmlib.Poisson:
						hmm.Obs[p][t*hmm.NState+j] = genPoisson(hmm.Mean[ip+j])
					case hmmlib.Tweedie:
						hmm.Obs[p][t*hmm.NState+j] = genTweedie(hmm.Mean[ip+j], 1, hmm.VarPower)
					default:
						panic("unknown model\n")
					}
				}
			}
		}
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

// GenStatesMulti creates a random state sequence in which
// there are no collisions.
func GenStatesMulti(hmm *hmmlib.MultiHMM) {

	row := make([]float64, hmm.NState)
	hmm.State = makeIntArray(hmm.NParticle, hmm.NTime)

	// Set the initial state
	for p := 0; p < hmm.NParticle; p++ {
		hmm.State[p][0] = genDiscrete(hmm.Init)
	}

	// Set the rest of the states
	for t := 1; t < hmm.NTime; t++ {

		for g := range hmm.Group {

			// Avoid collisions within a group at the same time
			check1 := make(map[int]bool)
			check2 := make(map[int]bool)

			q := len(hmm.Group[g])

			for i, p := range hmm.Group[g] {

				st0 := hmm.State[p][t-1]
				copy(row, hmm.Trans[st0*hmm.NState:(st0+1)*hmm.NState])

				for k := 0; ; k++ {
					st := genDiscrete(row)

					if i < q/2 && check1[st] {
						continue
					} else if check2[st] {
						continue
					}

					hmm.State[p][t] = st

					if i < q/2 {
						check1[st] = true
					} else {
						check2[st] = true
					}
					break
				}
			}
		}
	}

	// Mask
	for p := 0; p < hmm.NParticle; p++ {
		entry := rand.Int() % 100
		for t := 0; t < entry; t++ {
			hmm.State[p][t] = hmmlib.NullState
		}

		exit := 900 + (rand.Int() % 100)
		for t := exit; t < hmm.NTime; t++ {
			hmm.State[p][t] = hmmlib.NullState
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

// Generate a Gamma random variable with mean alp*bet and variance alp*bet^2.
// Based on gsl_ran_gamma
// https://raw.githubusercontent.com/ampl/gsl/master/randist/gamma.c
func genGamma(alp, bet float64) float64 {

	if alp < 1 {
		u := rand.Float64()
		return genGamma(1+alp, bet) * math.Pow(u, 1/alp)
	}
	d := alp - 1.0/3.0
	c := (1.0 / 3.0) / math.Sqrt(d)

	var v, x float64
	for {
		for {
			x = rand.NormFloat64()
			v = 1 + c*x
			if v > 0 {
				break
			}
		}

		v = v * v * v
		u := rand.Float64()
		if u < 1-0.0331*x*x*x*x {
			break
		}

		if math.Log(u) < 0.5*x*x+d*(1-v+math.Log(v)) {
			break
		}
	}

	return bet * d * v
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

// GenStates generates a random state sequence.  See GenStatesMulti for a version
// of this function that avoids collisions.
func GenStates(hmm *hmmlib.HMM) {

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
