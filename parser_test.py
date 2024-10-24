from data_utils.flight_data import * 
import os

# iff_path = Path('data/IFF_USA_20200101_050001_86397_LADDfiltered.csv')
# iff_path = iff_path.resolve()

base_stem = 'USA_20200101_050001_86397'
data_dir = Path('data/flight').resolve()

fd = FlightData(data_dir, base_stem)

# fd.make_trackpoints_csv()
# fd.make_merged_cleaned_csv()
fd.make_resampled_csv()

