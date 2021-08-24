# pyIsoFit
***
## What does it do?
`pyIsoFit` is an isotherm fitting package with co-adsorption prediction functionality using the extended dual-site Langmuir model.


pyIsoFit was created with a focus on fitting with isotherm models that exhibit thermodynamically correct behaviour. It is flexible and can fit any number of datasets with features such as an ability to toggle fitting constraints for the different fitting procedures implemented. Additionally, the package can be used to fit any number of isotherm data and readily generate plots of the fittings and tables of the fitting parameters for the user and predict co-adsorption using the extended dual-site Langmuir model. While currently the package is limited to only one co-adsorption fitting procedure, it sets foundational work for an extended model-based python package for multi-component prediction that might serve as an alternative to IAST.

Below is a summary of pyIsoFit's features:

- Fitting to 10 analyitcal isotherm models: Dubinin-Radushkevich (MDR), Guggenheim-Anderson-de Boer (GAB), Do and Do (DoDo), Brunauer, Deming, Deming and Teller (BDDT), Brunauer–Emmett–Teller (BET), Henry, Sips and Toth.
- Thermodynamically consistent fitting procedures for Langmuir and dual-site Langmuir (DSL).
- Co-adsorption prediction using extended DSL for any number of components.
- Heat of adsorption caculation for Langmuir, DSL and GAB
- Tabulating results, generates plots and saves them as .csv files.

## Installation

Latest stable release:

```bash
pip install pyIsoFit
```

## Usage
For a tutorial on how to use the package with examples and use cases please refer to the [pyIsoFit_demo.py](https://github.com/dominikpantak/pyIsoFit/blob/main/demo/pyIsofit_demo.ipynb) file.

