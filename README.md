# Simple chemical model for sprites

## Description
This code implements the chemical model described in the paper "Associative electron detachment in sprites" by A. Malagón-Romero _et al._ (under review).

## Requirements
The code requres the [CHEMISE python library](https://gitlab.com/aluque/chemise) that can be installed 
with
```
pip install chemise
```

NumPy and SciPy are also required.

## Use
```
usage: chemical.py [-h] [--model {rm78,current}] [--field FIELD [FIELD ...]] [--tend TEND] [--latex]

options:
  -h, --help            show this help message and exit
  --model {rm78,current}, -m {rm78,current}
                        Dissociative attachment model
  --field FIELD [FIELD ...], -e FIELD [FIELD ...]
                        Reduced field in Td
  --tend TEND, -t TEND  Final time in seconds
  --latex, -L           Produce latex output?
```

After running the model, the output is written to an `output` folder in a format that can be read by
[QtPlaskin](https://github.com/aluque/qtplaskin) (see [here](https://github.com/erwanp/qtplaskin) for an alternative fork).

