
import csv, os,sys
import pymysql.cursors

capstoneHost = os.getenv('capstoneHost')
capstoneUser = os.getenv('capstoneUser')
capstonePassword = os.getenv('capstonePassword')
capstoneDb = os.getenv('capstoneDb')
# Connect to the database
connection = pymysql.connect(host=capstoneHost,
                        user=capstoneUser,
                        password=capstonePassword,
                        db=capstoneDb)

try:
    with connection.cursor() as cursor:
        for file in os.listdir("/crunchbase_2013_snapshot_20131212"):
            if file.endswith(".sql"):
                table_name = file.replace('.sql', "")
                sql = "SELECT * FROM " + table_name + ";"
                cursor.execute(sql)
                result = cursor.fetchall()
                filepath = "data/" + table_name +".csv"
                myFile = csv.writer(open(filepath, 'w'))
                field_names = [i[0] for i in cursor.description]
                myFile.writerow(field_names)
                myFile.writerows(result)

finally:
    connection.close()