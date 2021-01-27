# EIT Data Analysis

This repository holds the complementary code for the paper `Towards More Robust Hand Gesture Recognition on EIT Data`, David P. Leins, Christian Gibas, Rainer Brück and Robert Haschke.

## Prerequisites

- Python 3.x.x
- pandas (1.1.2)
- numpy (1.19.1)
- matplotlib (3.3.0)
- seaborn (0.10.1)
- scikit-learn (0.23.1)

#### For Train and Plot

- tensorflow 2.x

run `pip install -r requirements_train.txt` to install all needed dependencies.

#### For Interactive Visualization

- plotly (4.12.0)
- chart-studio (1.1.0)
- ipywidgets (7.5.1)

run `pip install -r requirements_interactive.txt` to install all needed dependencies.

## Instructions

Download the dataset from [https://doi.org/10.4119/unibi/2948441](https://doi.org/10.4119/unibi/2948441) and extract the archive content into the `data` directory.

## Train and Plot

The [jupyter notebook](train_and_plot.ipynb) can be run to train any TensorFlow model on the raw data and the calibrated variants presented in the paper. The best performing models in the paper are defined in the define_auxiliary_functions script, which is loaded within the notebook.

Loss, optimizer, batch size, epochs, and repetitions can be changed easily by setting the respective variables defined right before the train loop.
To reproduce results from the paper, change epochs to 100 and repetitions to 5 or higher (for statistical relevance).

To train your own models, define a function returning a TensorFlow/Keras Model class and change the `models_to_train` list accordingly.

Below the training block you'll find a few plot suggestions including our current baseline accuracies.

<sub>Note that the code can easily be changed to train PyTorch models.</sub>

## Interactive Visualization

Run the [jupyter notebook](interactive_calibration_view.ipynb) to get an interactive visualization of global and local calibration on PCA or t-SNE 2D projections powered by plotly and ipywidgets.
This visualization can be filtered by session UID, class label and iteration.

Preview:
![[./resources/preview.png]]

<sub>Note that the visualization of many data points may be slow, depending on your browser and machine. Also changing colorization or switching between projections requires that all traces are
replotted and thus may take a while, too.</sub>

# Contact

Don't hesitate to open an Issue here or contact me (David Leins) via email under dleins@techfak.de

# Citations

coming soon

If you use the dataset in your research, please cite it with:

```
@misc{Leins2020EIT,
  author          = {D. {Leins} and C. {Gibas} and R. {Brück} and R. {Haschke}},
  keywords        = {EIT, Gesture Recognition},
  publisher       = {University of Siegen},
  title           = {{Hand Gesture Recognition with Electrical Impedance
                  Tomography (Dataset)}},
  url             = {https://pub.uni-bielefeld.de/record/2948441},
  doi             = {10.4119/unibi/2948441},
  year            = 2020,
}
```
