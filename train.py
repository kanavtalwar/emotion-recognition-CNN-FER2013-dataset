from model.model import Model
from data.data_preprocess import Dataset


if __name__ == "__main__":
    dataset = Dataset()
    print('Preprocessing Data..')
    dataset.get_data()
    model = Model()
    model.build_model()
    model.train(dataset.images, dataset.labels)