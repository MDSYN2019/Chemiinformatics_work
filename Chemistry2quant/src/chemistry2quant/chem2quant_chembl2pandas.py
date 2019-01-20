"""

Python module which returns the chembl database as a pandas table, depending on the 
specifications you would like in the end. 

Make sure that the presmissions settings on the table has been granted to the USER.

Usually, this would be 

"""
try:
    import psycopg2
except:
    print("Error: You need psycopg2 to run this code")
    
class chemblConnect:
    def __init__(self, database, user, host, password)
        self.database = str(database)
        self.user = str(user)
        self.host = str(host)
        self.password = str(password) 
        self.string = "dbname={} user={} host={} password={}".format(self.database, self.user, self.host, self.password)
        self.conn = psycopg2.connect(self.string)
    def issue_command(self):
        pass

    
        
