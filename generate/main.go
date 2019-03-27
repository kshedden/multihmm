package main

import (
	"compress/gzip"
	"encoding/gob"
	"math/rand"
	"os"

	"github.com/kshedden/multihmm/hmmlib"
)

const (
	nparticle = 10
	ntime     = 1000
	nstate    = 20

	obsmodel hmmlib.ObsModelType = hmmlib.Tweedie //hmmlib.Poisson //
)

func main() {

	rand.Seed(52424399)

	hmm := hmmlib.NewMulti(nparticle, nstate, ntime)
	hmm.ObsModelForm = obsmodel
	hmm.Aux = []float64{1.2}

	hmm.Trans = make([]float64, nstate*nstate)
	for i := 0; i < nstate; i++ {
		p := 0.4 + 0.4*float64(i)/float64(nstate-1)
		for j := 0; j < nstate; j++ {
			if i == j {
				hmm.Trans[i*nstate+j] = p
			} else {
				hmm.Trans[i*nstate+j] = (1 - p) / float64(nstate-1)
			}
		}
	}

	hmm.Init = make([]float64, nstate)
	for i := 0; i < hmm.NState; i++ {
		hmm.Init[i] = 1 / float64(nstate)
	}

	hmm.Mean = make([]float64, nstate*nstate)
	for i := 0; i < nstate; i++ {
		for j := 0; j < nstate; j++ {
			switch obsmodel {
			case hmmlib.Gaussian:
				if i == j {
					hmm.Mean[i*nstate+j] = 1
				}
			case hmmlib.Poisson:
				if i == j {
					hmm.Mean[i*nstate+j] = 8
				} else {
					hmm.Mean[i*nstate+j] = 1
				}
			case hmmlib.Tweedie:
				if i == j {
					hmm.Mean[i*nstate+j] = 8
				} else {
					hmm.Mean[i*nstate+j] = 1
				}
			default:
				panic("unkown obsmodel\n")
			}
		}
	}

	if hmm.ObsModelForm == hmmlib.Gaussian {
		hmm.Std = make([]float64, nstate*nstate)
		for i := 0; i < nstate*nstate; i++ {
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
