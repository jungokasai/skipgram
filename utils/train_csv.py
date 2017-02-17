from data_process import Dataset
import csv
from ast import literal_eval

if __name__ == '__main__':
    class Opts(object):
        def __init__(self):
            self.window_size = 10
            self.vocab_size = 400000
    opts = Opts()
    data = Dataset(opts)
    incomplete = data.next_batch(100)

    with open('train_x.csv', 'wb') as csvfile:
        train_x_writer=csv.writer(csvfile, delimiter=',')
        with open('train_y.csv', 'wb') as csvfile:
            train_y_writer=csv.writer(csvfile, delimiter=',')
            incomplete = data.next_batch(1)
            while incomplete:
                train_x_writer.writerow(data.X_train_batch)
                train_y_writer.writerow(data.y_train_batch)
                incomplete = data.next_batch(1)
    with open('train_x.csv', 'rb') as csvfile:
        train_x_reader = csv.reader(csvfile, delimiter=',')
        batch=[]
        for i in xrange(100):
            batch.append(int(train_x_reader.next()[0]))
        print(batch)
        
        

