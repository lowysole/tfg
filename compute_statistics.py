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
    parser.add_argument('test_file', help='Path from Labels file')
    parser.add_argument('output_file', help='Path to save the parsed file')
    args = parser.parse_args()

    testfile = open(args.test_file, 'r')
    test_reader = csv.DictReader(testfile)

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
                row['hasbird'] = None
                row['datasetid'] = None
                for test in test_reader:
                    if row['itemid'] == test['itemid']:
                        if test['hasbird'] == '1':
                            row['hasbird'] = True
                        else:
                            row['hasbird'] = False
                        row['datasetid'] = test['datasetid']
                        break
                if row['datasetid'] == None:
                    break
                prediction = float(row['prediction'])
                if prediction >= 0.5:
                    row['result'] = True
                else:
                    row['result'] = False

                if (row['hasbird'] and row['result']) or (
                    not row['hasbird'] and not row['result']):
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

    testfile.close()


if __name__ == "__main__":
    main()


