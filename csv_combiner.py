import csv

csv_file_paths = [
    r'C:\Users\joehi\Desktop\MPU\gesture_data_hammering_115200.csv',
    r'C:\Users\joehi\Desktop\MPU\gesture_data_sawing_115200.csv',
    r'C:\Users\joehi\Desktop\MPU\gesture_data_screwing_115200.csv',
    r'C:\Users\joehi\Desktop\MPU\gesture_data_inactive_115200.csv',
]

def combine_csvs():
    with open('combined_gesture_data_115200.csv', 'w', newline='') as outfile:
        # create a csv writer
        csv_writer = csv.writer(outfile)
        
        # read the column headers from the first csv file
        with open(csv_file_paths[0], 'r') as infile:
            csv_reader = csv.reader(infile)
            column_headers = next(csv_reader)
        
        # write the column headers to the output file
        csv_writer.writerow(column_headers)
        
        # iterate over the csv files
        for csv_file_path in csv_file_paths:
            with open(csv_file_path, 'r') as infile:
                # create a csv reader
                csv_reader = csv.reader(infile)
                
                # skip the first row (column headers)
                next(csv_reader)
                
                # write the remaining rows to the output file
                for row in csv_reader:
                    csv_writer.writerow(row)

if __name__ == '__main__':
    combine_csvs()


