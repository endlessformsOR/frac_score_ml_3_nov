# quickly view dataset

import os, io, pytz, time, pickle
import numpy as np
from bokeh.layouts import column
from bokeh.io import push_notebook, show, output_notebook
from bokeh.io import output_file
from bokeh.models import Button, TextAreaInput, LinearAxis, Range1d, Span, Dropdown, PreText, HoverTool
from ipywidgets import interact
from bokeh.models import CustomJS, Slider
from bokeh.plotting import ColumnDataSource, figure, output_file, show


cwd = os.getcwd()
print(cwd)

from ml_utils import print_out_well_job_info, mk_utc_time, fetch_sensor_ids_from_apis, mk_random_windows, make_data, \
    butter_bandpass_filter, make_fft_matrix

sampling_rate = 40000

output_notebook()

# set up bokeh plots, updates to follow
data_file_name = 'raw_ml_data/frac_score_ml_dataset.npy'
events = np.load(data_file_name, allow_pickle=True)

print('number of events: ', len(events))

# j = np.linspace(0,len(events)-1,len(events))
# print(j[-1])
# i = int(j[z])

z = np.linspace(0, 10, 500)
source = ColumnDataSource(data=dict(z=z))
data_slider = Slider(start=0, end=len(events), value=1, step=1, title="Event Number")
callback = CustomJS(args=dict(source=source, num=data_slider),
                              code="""
                              
                              const data = source.data;
                              const i = num.value;
                              
                              source.change.emit();
                              """)
data_slider.js_on_change('value',callback)

i = 454

well_name = events[i][0]
api = events[i][1]
reason = events[i][2]
start_date = events[i][3]
start_time = events[i][4]
end_date = events[i][5]
end_time = events[i][6]
stage = events[i][7]
utc_start = events[i][8]
comments = events[i][9]
dynamic_values = events[i][10]
new_label = events[i][11]

time_values = np.linspace(0, (1000 * len(dynamic_values) / sampling_rate), len(dynamic_values))  # in milliseconds

grad_raw = np.gradient(dynamic_values)

filtered_signal = butter_bandpass_filter(dynamic_values, 50, 500, sampling_rate, order=9)
max_dyn = np.nanmax(np.absolute(filtered_signal))
dynamic_normalized = np.absolute(filtered_signal) / np.absolute(max_dyn)
dynamic_gradient = np.gradient(dynamic_normalized)

# add metadata

textstr = 'WN: ' + well_name + '\n'
textstr += 'API: ' + api + '\n'
textstr += 'Stage: ' + stage + '\n'
textstr += 'Start: ' + utc_start + '\n'
# textstr += 'Stop: ' + stop_time + '\n'
textstr += 'index in dataset: ' + str(i) + '\n'

EventPreText = PreText(text='\n' + textstr + '\n',
                       width=500,
                       height=150)

# create bokeh objects

plot_height = 125
plot_width = 625
plot_tools = 'hover, box_zoom, reset'

p = figure(title="Dynamic Pressure Response",
           plot_height=plot_height,
           plot_width=plot_width,
           x_range=(np.nanmin(time_values), np.nanmax(time_values)),
           y_range=(np.nanmin(dynamic_values), np.nanmax(dynamic_values)),
           x_axis_label='time in milliseconds',
           y_axis_label='arb units',
           tools=plot_tools,
           background_fill_color='#efefef')

#p.legend.location = "top_left"
#p.legend.click_policy = "hide"

s = figure(title="Gradient: Raw",
           plot_height=plot_height,
           plot_width=plot_width,
           x_range=(np.nanmin(time_values), np.nanmax(time_values)),
           y_range=(np.nanmin(grad_raw), np.nanmax(grad_raw)),
           x_axis_label='time in milliseconds',
           y_axis_label='arb units',
           tools=plot_tools,
           background_fill_color='#efefef')

#s.legend.location = "top_left"
#s.legend.click_policy = "hide"

t = figure(title="spectro as image, bokeh",
           x_range=(0, 100),
           y_range=(0, 100))

# add lines

a = p.line(time_values,
           dynamic_values,
           color="black",
           line_width=1.5,
           alpha=0.8,
           legend_label='raw')

d = s.line(time_values,
           grad_raw,
           color="green",
           line_width=1.5,
           alpha=0.8,
           legend_label='gradient of raw dynamic signal')

# use this to interact with plots

# def update(qc_status, pop_status, i):
def update(i):
    """
    i - index of detected pop
    qc_status - Has a human looked at this?
    pop_status - Is this an actual pop, or not? Or something else?

    """
    well_name = events[i][0]
    api = events[i][1]
    reason = events[i][2]
    start_date = events[i][3]
    start_time = events[i][4]
    end_date = events[i][5]
    end_time = events[i][6]
    stage = events[i][7]
    utc_start = events[i][8]
    comments = events[i][9]
    dynamic_values = events[i][10]
    label = events[i][11]

    time_values = np.linspace(0, (1000 * len(dynamic_values) / sampling_rate), len(dynamic_values))  # in milliseconds

    grad_raw = np.gradient(dynamic_values)

    filtered_signal = butter_bandpass_filter(dynamic_values, 50, 500, sampling_rate, order=9)
    max_dyn = np.nanmax(np.absolute(filtered_signal))
    dynamic_normalized = np.absolute(filtered_signal) / np.absolute(max_dyn)
    dynamic_gradient = np.gradient(dynamic_normalized)
    #standard_scalar_obj = standard_scalar(dynamic_values)

    num_partitions = 40
    spectrogram_matrix = make_fft_matrix(dynamic_values, sampling_rate, 50)

    # add metadata

    if label == 0:
        label = 'NOT A POP'
    if label == 1:
        label = 'POP'

    textstr = 'WN: ' + well_name + '\n'
    textstr += 'API: ' + api + '\n'
    textstr += 'Stage: ' + stage + '\n'
    textstr += 'Start: ' + start_time + '\n'
    textstr += 'label: ' + label + '\n'
    textstr += 'index in dataset: ' + str(i) + '\n'

    EventPreText = PreText(text='\n' + textstr + '\n',
                           width=500,
                           height=150)

    # create bokeh objects

    plot_height = 250
    plot_width = 1100
    plot_tools = "hover,save,pan,box_zoom,reset,wheel_zoom"

    p = figure(title="Dynamic Pressure Response",
               plot_height=plot_height,
               plot_width=plot_width,
               x_range=(np.nanmin(time_values), np.nanmax(time_values)),
               y_range=(np.nanmin(dynamic_values), np.nanmax(dynamic_values)),
               x_axis_label='time in milliseconds',
               y_axis_label='arb units',
               tools=plot_tools,
               background_fill_color='#efefef')

    s = figure(title="Gradient: Raw",
               plot_height=plot_height,
               plot_width=plot_width,
               x_range=(np.nanmin(time_values), np.nanmax(time_values)),
               y_range=(np.nanmin(grad_raw), np.nanmax(grad_raw)),
               x_axis_label='time in milliseconds',
               y_axis_label='arb units',
               tools=plot_tools,
               background_fill_color='#efefef')

    t = figure(title="Image Spectrogram of Signal",
               plot_height=plot_height,
               plot_width=plot_width,
               x_range=(0, 2),
               y_range=(0, 2))

    # add lines

    a = p.line(time_values,
               dynamic_values,
               color="black",
               line_width=1.5,
               alpha=0.8,
               legend_label='raw')

    d = s.line(time_values,
               grad_raw,
               color="green",
               line_width=1.5,
               alpha=0.8,
               legend_label='gradient of raw dynamic signal')


    spectrogram_matrix = np.flip(spectrogram_matrix, axis=0)

    t.image(image=[spectrogram_matrix], x=0, y=0, dw=2, dh=2, palette="Spectral11")


    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    s.legend.location = "top_left"
    s.legend.click_policy = "hide"

    # update data sources

    a.data_source.data['dynamic_values'] = dynamic_values
    d.data_source.data['grad_raw'] = grad_raw
    #t.data_source.data['image'] = spectrogram_matrix

    push_notebook()

    layout = column(EventPreText, data_slider, p, s, t)
    output_file('data_viewer.html')
    show(layout)

#interact(update, i=(0, len(events) - 1))
interact(update, i=250)
