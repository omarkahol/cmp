# CMP 

CMP++ or CMP is a C++ library used to perform the Bayesian calibration of computer codes using many approaches such as

- Full Bayes
- Modular, sequential
- Modular, adaptive

You can find additional details in my personal [website](https://omarkahol.github.io/assets/pdf/projects/cmp/cmp.pdf).
## Getting Started

These instructions will help you compile and link the CMP++ library to your project. Please refer to the [technical documentation](https://omarkahol.github.io/assets/pdf/projects/cmp/cmp.pdf).

### Dependencies

The library depends on:

- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- [spdlog](https://github.com/gabime/spdlog)
- [NLopt](https://nlopt.readthedocs.io/en/latest/)

### Installing

CMP++ can be compiled into a static library using the provided makefile. Make sure to modify the paths for the dependencies

    cd cmp
    make

## Running the tests

Explain how to run the automated tests for this system

### Sample Tests

To run the test, do

    cd test_cmp
    make
    ./out

It will generate some csv files which can be processed using the plot.py file

    python3 plot.py

## License

This project is licensed under the [MIT license](LICENSE.md)
Creative Commons License - see the [LICENSE.md](LICENSE.md) file for
details

## Acknowledgments

This project has received funding from the European Union’s Horizon Europe research and innovation programme under the Marie Skłodowska-Curie grant agreement No 101072551 (TRACES).

Please refer to the [project website](https://traces-project.eu) for additional details.
