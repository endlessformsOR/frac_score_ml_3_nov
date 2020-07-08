
class dataLoadingClass():

    def __init__(self):

        import pandas as pd
        
        # DEFINE OUTPUT DATAFRAME COLS
        
        self.conditionedDataSet = pd.DataFrame(columns=['EVENT',
                                                        'START_TIME',
                                                        'STOP_TIME',
                                                        'CLASS',
                                                        'METADATA',
                                                        'TAGS',
                                                        'WELL_API',
                                                        'WELL_NAME',
                                                        'PAD',
                                                        'STAGE',
                                                        'STATIC_SENSOR_ID',
                                                        'DYNAMIC_SENSOR_ID',
                                                        'EFF_COMMENTS',
                                                        'EFF_EVENT_TYPE',
                                                        'INDEX_IN_UNC_DATA'])
        # DEFINE VARS, these will change
        
        self.eventIndex = 50
        self.eventType = '-999'
        self.startTime = '-999'
        self.endTime = '-999'
        self.Api = '-999'
        self.WN = '-999'
        self.staticID = '-999'
        self.dynamicID = '-999'
        self.metadata = '-999'
        self.tags = '-999'
        self.pad = '-999'
        self.stage = '-999'
        self.effComments = '-999'
        self.effEventType = '-999'
        self.classType = '-999'
        self.seconds = 1
        
        self.bolFirstPlt = True
        
        # Vars used for plotting 
        
        self.textstr = ''
        self.x_d = []
        self.t_d = []
        self.sampling_rate = 45000
        self.x_s = []
        self.t_s = []
        self.x_d_r = [] 
        self.t_d_r = []
        self.global_max_x_val = 0
        self.new_start_time_str = ''
        self.new_stop_time_str = ''
        
        self.tmin_s = -999
        self.tmax_s = -999
        self.pmin_s = -999
        self.pmax_s = -999

        self.tmin_d = -999
        self.tmax_d = -999 
        self.pmin_d = -999
        self.pmax_d = -999

        # SEND TO DATAFRAME
       
        self.conditionedDataSet = self.conditionedDataSet.append({'EVENT' : self.classType,
                                                                  'START_TIME': self.new_start_time_str,
                                                                  'STOP_TIME': self.new_stop_time_str,
                                                                  'CLASS' : self.classType,
                                                                  'METADATA' : self.metadata,
                                                                  'TAGS' : self.tags,
                                                                  'WELL_API': self.Api,
                                                                  'WELL_NAME': self.WN,
                                                                  'PAD': self.pad,
                                                                  'STAGE': self.stage,
                                                                  'STATIC_SENSOR_ID': self.staticID,
                                                                  'DYNAMIC_SENSOR_ID': self.dynamicID,
                                                                  'EFF_COMMENTS': self.effComments,
                                                                  'EFF_EVENT_TYPE': self.effEventType,
                                                                  'INDEX_IN_UNC_DATA': str(self.eventIndex)}, ignore_index=True)
        
        self.bolFirstPlt = False
        
        
# PULL DATA OBJECTS FROM DATAFRAME METHOD

    def pullDataObjsFromDF(self,df, i):

        self.eventType = str(df.loc[i,"EVENT_CLASS"])
        self.startTime = str(df.loc[i,"START_TIME"])
        self.endTime = str(df.loc[i, "STOP_TIME"])
        self.Api = str(df.loc[i, "WELL_API"])
        self.WN = str(df.loc[i,"WELL_NAME"])
        self.staticID = str(df.loc[i,"STATIC_SENSOR_ID"])
        self.dynamicID = str(df.loc[i,"DYNAMIC_SENSOR_ID"])

        self.effComments = ''

        if 'EFF_COMMENTS' in df.columns:

            self.effComments = str(df.loc[i,"EFF_COMMENTS"])

        if 'COMMENT' in df.columns:

            self.effComments = str(df.loc[i,"COMMENT"])

        self.stageNumber = ''

        if 'STAGE' in df.columns:

            self.stageNumber = str(df.loc[i,"STAGE"])

        # Well Info 

        self.textstr = 'Well Name: ' + self.WN + '\n' 
        self.textstr +='API: ' + self.Api + '\n' 
        self.textstr +='Event Type: ' + self.eventType + '\n' 
        self.textstr +='Start: ' + self.startTime + '\n' 
        self.textstr +='End: ' + self.endTime + '\n'
        self.textstr +='Stage Number: ' + self.stageNumber + '\n'
        self.textstr +='Comments: ' + self.effComments[:50] + '\n'
        self.textstr +='Comments: ' + self.effComments[50:100] + '\n'
        # make dynamic data

        print('downloading time series data for: ' + self.WN + '...')

        self.start_converted = parse_time_string_with_colon_offset(self.startTime)
        self.stop_converted = parse_time_string_with_colon_offset(self.endTime)
        self.delta = self.stop_converted - self.start_converted
        self.seconds = int(self.delta.total_seconds())

        self.x_d = interval_to_flat_array(self.dynamicID, self.start_converted, self.stop_converted).values()
        self.t_d = np.linspace(0,len(self.x_d),len(self.x_d))

        self.sampling_rate = int(len(self.x_d) / self.seconds)

        # make static data

        print('making static and dynamic data...')

        self.stat_obj = fetch_sensor_db_data(self.staticID,self.startTime,self.endTime)

        self.x_s = self.stat_obj['max'].to_numpy()
        self.t_s = np.linspace(0,len(self.x_s),len(self.x_s))

        return self.t_s, self.x_s, self.t_d, self.x_d, self.sampling_rate, self.seconds, self.textstr, self.startTime, self.endTime
        