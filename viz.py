import airportsdata
apd_iata = airportsdata.load('IATA')  # key is the IATA location code

# top 10 busiest southwest, iata codes
# busiest_iata = ['DEN', 'DAL', 'MDW', 'PHX', 'HOU', 'LAS', 'MCO', 'BNA', 'BWI', 'OAK']

busiest_iata = ['DEN', 'MDW', 'DAL', 'LAS']
# zip codes: 80249, 60638, 75235, 89119
# https://www.itl.nist.gov/div898/winds/asos-wx/WBAN-MSC.TXT wban codes

southwest_id = 'WN'

# cols = ['Quarter', 'Year', 'Month', 'DayofMonth', 'DayOfWeek', 'ReportingAirline', 'ArrDel15', 'DepDel15']

