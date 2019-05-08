package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"os/exec"
	"path"
	"regexp"
	"strconv"
	"strings"
)

const (
	ngroup       = 20
	nparticlegrp = 10
	nstate       = 20
	ntime        = 540
	snr          = 3.0
)

var (
	logger *log.Logger

	out io.WriteCloser
)

func generate(pw float64) {

	c := []string{"run", "../generate/main.go",
		fmt.Sprintf("-obsmodel=%s", "tweedie"),
		fmt.Sprintf("-ngroup=%d", ngroup),
		fmt.Sprintf("-varpower=%f", pw),
		fmt.Sprintf("-nparticlegrp=%d", nparticlegrp),
		fmt.Sprintf("-ntime=%d", ntime),
		fmt.Sprintf("-nstate=%d", nstate),
		fmt.Sprintf("-outname=%s", "tweedie_tmp.gob.gz"),
		fmt.Sprintf("-snr=%f", snr),
	}

	logger.Printf("go %s\n", strings.Join(c, " "))

	cmd := exec.Command("go", c...)
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		panic(err)
	}
}

func fit(num int, pwt float64) string {

	logname := path.Join("tweedie_logs", fmt.Sprintf("%d_%d", num, int(10*pwt)))

	c := []string{"run", "../estimate/main.go",
		fmt.Sprintf("-maxiter=%d", 100),
		fmt.Sprintf("-logname=%s", logname),
		fmt.Sprintf("-varpower=%f", pwt),
		fmt.Sprintf("-gobfile=%s", "tweedie_tmp.gob.gz"),
		fmt.Sprintf("-reconstruct=%s", "false"),
	}

	logger.Printf("go %s\n", strings.Join(c, " "))
	cmd := exec.Command("go", c...)
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		panic(err)
	}

	return logname
}

func collect(logname string) float64 {

	fid, err := os.Open(logname + "_msg.log")
	if err != nil {
		panic(err)
	}
	defer fid.Close()

	scanner := bufio.NewScanner(fid)

	ec := regexp.MustCompile(`Final AIC: ([\d-.]*)`)

	for scanner.Scan() {

		line := scanner.Text()

		ma := ec.FindAllSubmatch([]byte(line), -1)
		if len(ma) == 0 {
			continue
		}

		llfv, err := strconv.ParseFloat(string(ma[0][1]), 64)
		if err != nil {
			panic(err)
		}
		return llfv
	}

	return math.NaN()
}

func run() {

	pwl := []float64{1.1, 1.3, 1.5, 1.7, 1.9}
	ii := 0

	for _, pwg := range pwl {
		for i := 0; i < 5; i++ {
			generate(pwg)
			for _, pwt := range pwl {
				logname := fit(i, pwt)
				llf := collect(logname)
				println(llf)
				_, _ = io.WriteString(out, fmt.Sprintf("%d,%f,%f,%f\n", ii, pwg, pwt, llf))
			}
			ii++
		}
	}
}

func main() {

	var err error
	out, err = os.Create("tweedie_result.csv")
	if err != nil {
		panic(err)
	}
	defer out.Close()

	head := "Run,TrueVarPower,VarPower,llf\n"
	_, _ = io.WriteString(out, head)

	// Logger for this script (every estimate run has its own log)
	lfid, err := os.Create("sim_tweedie.log")
	if err != nil {
		panic(err)
	}
	defer lfid.Close()
	logger = log.New(lfid, "", log.Ltime)

	run()
}
