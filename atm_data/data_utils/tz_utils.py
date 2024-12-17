import timezonefinder as tzf
import airportsdata as apd

def get_tz(lat, lon):
    tf = tzf.TimezoneFinder()
    timezone_str = tf.certain_timezone_at(lat=lat, lng=lon)
    if timezone_str is None:
        raise ValueError("Could not determine the time zone")
    return timezone_str


apd_iata = apd.load('IATA')

def get_tz_airport_iata(code):
    ap = apd_iata.get(code)
    return get_tz(ap['lat'], ap['lon'])