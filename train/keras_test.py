from get_model import get_model
from config_reader import config_test_reader


if __name__ == '__main__':

    params_test,params_model = config_test_reader()
    model = get_model(params_test,params_model)
    model.load_weights(params_model['keras_model_weights'])

    print (model.summary())