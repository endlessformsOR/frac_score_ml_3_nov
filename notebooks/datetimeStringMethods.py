# methods developed to deal with datetime stuff

import pytz
from datetime import datetime, timedelta

def parse_time_string_with_colon_offset(s):
    "parses timestamp string with colon in UTC offset ex: -6:00"
    if ":" == s[-3:-2]:
        s = s[:-3] + s[-2:]
    local = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")
    utc = local.astimezone(pytz.utc)
    return utc

def mkDataTimeFromStr(s):
    "same as parse_time_string, but includes utc offset"
    if ":" == s[-3:-2]:
        s = s[:-3] + s[-2:]
    local = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")
    #utc = local.astimezone(pytz.utc)
    #return utc
    return local

def dtObj2str(dtobj):
    "takes datetimeobj and returns formatted string with UTC offset"
    nst_formatted = dtobj.strftime("%Y-%m-%dT%H:%M:%S%z")
    # add the ":" to the format (couldn't find an easier solution)
    str_time = nst_formatted[:-5]
    str_utc = nst_formatted[-4:]
    str_utcUp = nst_formatted[-4:-2] + ':' + nst_formatted[-2:]
    outputSTR = str_time + '-'+ str_utcUp
    return outputSTR