- `flight_data_test.py`: driver to test `data_utils/flight_data_utils.py`, some stuff for cleaning and processing already downloaded nasa sherlock trajectory data in IFF/RD/EV files, hasn't been updated in a while
- `schedule_data_test.py`: currently has some stuff i was using to process https://developer.ibm.com/exchanges/data/all/airline/ airline flight schedule dataset, need to generalize and move into `data_utils/schedule_data_utils.py` to make general reusable methods like for the noaa weather
- `weather_noaa_data_test.py`: driver to test `data_utils/weather_noaa_data_utils.py`, like methods to download and process weather data from LCDv2 and GHCNh datasets
- `merged_data_test.py`: a bit messy right now but operates on the cleaned schedule/weather data and merges it together, to augment schedule data with the weather at departure airport at departure time, and weather at arrival airport at arrival time, for each flight, and generates all the plots in `media/`
- will add documentation for the utils stuff in a bit


atm is probably more accurate for what we're interested in not atc in the repo name 

```
export CFLAGS="-I $(brew --prefix graphviz)/include"
export LDFLAGS="-L $(brew --prefix graphviz)/lib"
poetry add pygraphviz
```
