from ds_helpers.io_helpers import load_classifier_predictions

def test_load_classifier_predictions():

    df = load_classifier_predictions()

    assert df.shape == (1000,2), "Wrong data shape, check size of table"