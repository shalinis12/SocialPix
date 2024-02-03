import pyodbc 

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=shalini;'
                      'Database=SocialMedia;'
                      'Trusted_Connection=yes;')

cursor = conn.cursor()
cursor.execute('SELECT * FROM Categ')

for i in cursor:
    print(i)


cursor.execute('''
                INSERT INTO Categ (Food, People)
                VALUES
                ('IceCream','Hanna'),
                ('Cookies','Tanya')
                ''')
conn.commit()