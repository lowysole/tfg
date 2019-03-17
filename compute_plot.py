import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Path from file to plot')
    parser.add_argument('output_name', help='Name of the plot')
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        reader = csv.DictReader(f)
        epoch = []
        acc = []
        loss = []
        val_acc = []
        val_loss = []
        for row in reader:
            epoch.append(float(row['epoch']))
            acc.append(float(row['acc']))
            loss.append(float(row['loss']))
            val_acc.append(float(row['val_acc']))
            val_loss.append(float(row['val_loss']))
    #Plot Accuracity
    plt.plot(epoch,acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracity')
    plt.grid(True)
    plt.savefig(args.output_name + "_acc.png")
    plt.show()

    #Plot Loss
    plt.plot(epoch,loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(args.output_name + "_loss.png")
    plt.show()

    #Plot Eval Accuracity
    plt.plot(epoch,val_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracity')
    plt.grid(True)
    plt.savefig(args.output_name + "_val_acc.png")
    plt.show()

    #Plot Loss
    plt.plot(epoch,val_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.grid(True)
    plt.savefig(args.output_name + "_val_loss.png")
    plt.show()





if __name__ == "__main__":
    main()


