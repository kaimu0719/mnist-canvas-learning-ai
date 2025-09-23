import pickle
import os

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

def load_mnist():
  with open(save_file, "rb") as file:
    dataset = pickle.load(file)

  return (dataset["train_img"], dataset["train_label"], dataset["test_img"], dataset["test_label"])