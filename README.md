# ML Project - Real Estate Price Prediction

Contributers: @jonaw1, @Kaesekuchen5, @VenthanV

## How to make predictions?

- Create file(s) `data/production/<your_file>.csv`
- The files should contain all columns according to `data/data_description.txt` or `data/data_description.json` minus the target column `Verkaufspreis`
- Run `main.py` and make sure to keep functions `preprocess()` and `train_model()` commented out
- Predictions can now be found in `predicted/<your_file>_predicted.json`

## Nice to know

- As long as there are results for a file in `predictions/`, the model will not re-predict, even if there were changes made to the input file

## Possible improvements
- Most impactful improvements could probably be achieved through feature engineering and/or selecting a different regressor model, because random forests don't work very well with one-hot encoding