package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"

	"github.com/kshedden/multihmm/hmmlib"
)

var (
	logger *log.Logger

	obsmodelnames = map[hmmlib.ObsModelType]string{
		hmmlib.Gaussian: "gaussian",
		hmmlib.Poisson:  "poisson",
		hmmlib.Tweedie:  "tweedie",
	}

	varmodelnames = map[hmmlib.VarModelType]string{
		hmmlib.VarConst: "varconst",
		hmmlib.VarFree:  "varfree",
	}
)

type model struct {
	obsmodel     hmmlib.ObsModelType
	varmodel     hmmlib.VarModelType
	zeroinflated bool
	varpower     float64
	ngroup       int
	nparticlegrp int
	ntime        int
	nstate       int
	gobfile      string
	logname      string
	maxiter      int
	nkp          int
}

var (
	basemodel *model = &model{
		obsmodel:     hmmlib.Gaussian,
		varmodel:     hmmlib.VarConst,
		zeroinflated: false,
		varpower:     1.5,
		ngroup:       20,
		nparticlegrp: 10,
		ntime:        1000,
		nstate:       20,
		gobfile:      "tmp.gob.gz",
		logname:      "hmm",
		maxiter:      20,
		nkp:          200,
	}
)

func generate(g *model) {

	c := []string{"run", "../generate/main.go",
		fmt.Sprintf("-obsmodel=%s", obsmodelnames[g.obsmodel]),
		fmt.Sprintf("-varmodel=%s", varmodelnames[g.varmodel]),
		fmt.Sprintf("-ngroup=%d", g.ngroup),
		fmt.Sprintf("-varpower=%f", g.varpower),
		fmt.Sprintf("-zeroinflated=%t", g.zeroinflated),
		fmt.Sprintf("-nparticlegrp=%d", g.nparticlegrp),
		fmt.Sprintf("-ntime=%d", g.ntime),
		fmt.Sprintf("-nstate=%d", g.nstate),
		fmt.Sprintf("-outname=%s", g.gobfile),
	}

	logger.Printf("go %s\n", strings.Join(c, " "))

	cmd := exec.Command("go", c...)
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		panic(err)
	}
}

func fit(g *model) {

	c := []string{"run", "../estimate/main.go",
		fmt.Sprintf("-maxiter=%d", g.maxiter),
		fmt.Sprintf("-nkp=%d", g.nkp),
		fmt.Sprintf("-logname=%s", g.logname),
		fmt.Sprintf("-gobfile=%s", g.gobfile),
	}

	logger.Printf("go %s\n", strings.Join(c, " "))
	cmd := exec.Command("go", c...)
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		panic(err)
	}
}

func run(m *model) {

	m.obsmodel = hmmlib.Tweedie
	m.zeroinflated = false
	m.varpower = 1.5

	m.nkp = 200
	for i := 0; i < 1; i++ {
		generate(m)
		fit(m)
		m.nkp += 20
	}
}

func main() {

	lfid, err := os.Create("sim.log")
	if err != nil {
		panic(err)
	}
	defer lfid.Close()
	logger = log.New(lfid, "", log.Ltime)

	run(basemodel)
}
