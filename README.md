# Gender Prediction ML Model

This machine learning project implements a simple gender prediction model based on physical characteristics (height and weight). The model uses a Decision Tree Classifier to predict whether a person is male or female based on their physical measurements.

## Features

- Predicts gender (Male/Female) based on:
  - Height (cm)
  - Weight (kg)
- Uses scikit-learn's Decision Tree Classifier
- Interactive command-line interface for real-time predictions
- Built with Python and popular data science libraries

## Requirements

- Python 3.x
- NumPy
- Pandas
- scikit-learn

## Installation

1. Clone this repository:
```bash
git clone [your-repository-url]
```

2. Install the required packages:
```bash
pip install numpy pandas scikit-learn
```

## Usage

1. Run the application:
```bash
python app.py
```

2. When prompted:
   - Enter the height (in centimeters)
   - Enter the weight (in kilograms)

The model will then predict the gender based on the input values.

## Model Details

- Algorithm: Decision Tree Classifier
- Features: Height and Weight
- Target Variable: Sex (M/W)
- Train-Test Split: 80-20
- Random State: 42

## Dataset

The model is trained on a sample dataset containing the following features:
- Height (cm)
- Weight (kg)
- Age (years)
- Sex (M/W)

## Future Improvements

- Add more features for better prediction accuracy
- Implement cross-validation
- Add data visualization
- Expand the training dataset
- Try different machine learning algorithms

## License

This project is open source and available under the MIT License.

## Contributing

Contributions, issues, and feature requests are welcome! 