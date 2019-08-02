import csv
from subprocess import call

def download_data(data_path, out_path):
    with open(data_path) as csvfile:
        list_reader = csv.reader(csvfile)
        data = [r for r in list_reader]
        data.pop(0)
        for img_info in data:
            # wget.download(img_info[0], out=out_path)
            call(["wget", img_info[0], "-t", "5", "-T", "60", "-P", out_path])

if __name__ == '__main__':
    download_data("../data/test/defect_testing_gt.csv", "../dataset/test")
    download_data("../data/train/defect_training_gt.csv", "../dataset/train")
