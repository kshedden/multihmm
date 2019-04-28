package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path"
	"regexp"
	"strconv"
	"strings"

	"github.com/kshedden/multihmm/hmmlib"
)

var (
	logger *log.Logger

	out io.WriteCloser

	obsmodelnames = map[hmmlib.ObsModelType]string{
		hmmlib.Gaussian: "gaussian",
		hmmlib.Poisson:  "poisson",
		hmmlib.Tweedie:  "tweedie",
	}

	varmodelnames = map[hmmlib.VarModelType]string{
		hmmlib.VarConst: "varconst",
		hmmlib.VarFree:  "varfree",
	}

	constraint hmmlib.ConstraintMaker
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
	snr          float64
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
		snr:          4,
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
		fmt.Sprintf("-snr=%f", g.snr),
	}

	logger.Printf("go %s\n", strings.Join(c, " "))

	cmd := exec.Command("go", c...)
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		panic(err)
	}
}

func fit(g *model, num int) {

	logname := fmt.Sprintf("%s_%d", obsmodelnames[g.obsmodel], num)

	c := []string{"run", "../estimate/main.go",
		fmt.Sprintf("-maxiter=%d", g.maxiter),
		fmt.Sprintf("-nkp=%d", g.nkp),
		fmt.Sprintf("-logname=%s", path.Join("logs", logname)),
		fmt.Sprintf("-constraint=%s", "flexcollision"),
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

func collect() []int {

	fid, err := os.Open("hmm_msg.log")
	if err != nil {
		panic(err)
	}
	defer fid.Close()

	scanner := bufio.NewScanner(fid)

	ec := regexp.MustCompile(`(\d*)/(\d*) total errors`)

	var nr []int

	for scanner.Scan() {

		line := scanner.Text()

		ma := ec.FindAllSubmatch([]byte(line), -1)
		if len(ma) == 0 {
			continue
		}
		numer, err := strconv.Atoi(string(ma[0][1]))
		if err != nil {
			panic(err)
		}

		denom, err := strconv.Atoi(string(ma[0][2]))
		if err != nil {
			panic(err)
		}

		nr = append(nr, numer, denom)
	}

	return nr
}

func run(m *model) {

	mna := obsmodelnames[m.obsmodel]
	vmna := varmodelnames[m.varmodel]

	for i := 0; i < 10; i++ {
		generate(m)

		for _, m.nkp = range []int{2, 5, 10, 20, 30, 40} {
			fit(m, i)
			nr := collect()
			_, _ = io.WriteString(out, fmt.Sprintf("%s,%s,%.2f,%d,%d,%d,%d,%d\n",
				mna, vmna, m.varpower, m.nkp, i, nr[0], nr[2], nr[3]))
		}
	}
}

func main() {

	var err error
	out, err = os.Create("result.csv")
	if err != nil {
		panic(err)
	}
	defer out.Close()

	head := "EmissionModel,VarModel,VarPower,nkp,Run,ErrMarg,ErrJoint,Total\n"
	_, _ = io.WriteString(out, head)

	lfid, err := os.Create("sim.log")
	if err != nil {
		panic(err)
	}
	defer lfid.Close()
	logger = log.New(lfid, "", log.Ltime)

	m := basemodel
	m.obsmodel = hmmlib.Poisson
	run(basemodel)

	m = basemodel
	m.obsmodel = hmmlib.Gaussian
	run(basemodel)

	m.obsmodel = hmmlib.Tweedie
	for _, vp := range []float64{1.2, 1.4, 1.6, 1.8} {
		m.varpower = vp
		run(basemodel)
	}
}
