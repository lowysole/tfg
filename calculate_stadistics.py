import csv
import argparse

HEADER = ['itemid',
          'prediction',
          'datasetid',
          'hasbird',
          'result'
          ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Path from file to parser')
    parser.add_argument('output_file', help='Path to save the parsed file')
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        import pudb ; pudb.set_trace()
        reader = csv.DictReader(f)
        with open(args.output_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames = HEADER)
            writer.writeheader()
            count = 0
            count_true = 0
            count_false = 0
            for row in reader:
                prediction = float(row['prediction'])
                if prediction >= 0.5:
                    row['result'] = 'True'
                else:
                    row['result'] = 'False'

                if ((row['hasbird'] == 'True' and prediction >= 0.5) or (
                row['hasbird'] == 'False' and prediction < 0.5)):
                    count_true = count_true + 1
                else:
                    count_false = count_false +1
                count = count + 1
                writer.writerow(row)


            print('Count: {}'.format(count))
            print('Correct: {}'.format(count_true))
            print('Correct % : {}'.format(count_true/count))
            print('Incorrect: {}'.format(count_false))
            print('Incorrect % : {}'.format(count_false/count))


if __name__ == "__main__":
    main()


