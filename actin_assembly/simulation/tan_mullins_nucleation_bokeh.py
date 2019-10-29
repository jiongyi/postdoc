from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider
from bokeh.plotting import figure
from numpy import sqrt, linspace

def steady_state(load_const = 1.0, transfer_const = 0.1,
               elongation_const = 0.5, nucleation_const = 0.05):
    # Maximum capping rate constant.
    max_capping_const = nucleation_const * 1.0**2
    capping_rate_row = linspace(1e-3, max_capping_const, 100)
    # Steady state for WA domain occupancy.
    wh2_eq_row = sqrt(capping_rate_row / nucleation_const)
    # Set up quadratic.
    a = elongation_const * (capping_rate_row + elongation_const)
    b = (load_const + transfer_const * (1.0 - wh2_eq_row)) * (capping_rate_row + elongation_const)
    c = - load_const * transfer_const * (1.0 - wh2_eq_row)
    ends_eq_row = (-b + sqrt(b**2 - 4 * a * c)) / (2 * a)
    ppr_eq_row = load_const / (load_const + elongation_const * ends_eq_row + transfer_const * (1.0 - wh2_eq_row))
    return ppr_eq_row, wh2_eq_row, ends_eq_row

# Set up x-y data.
init_nucleation_const = 1.0
init_capping_rate_row = linspace(0.1, init_nucleation_const, 100)

init_ppr_eq_row, init_wh2_eq_row, init_ends_eq_row = steady_state2(nucleation_const = init_nucleation_const)
data_source = ColumnDataSource(data = dict(x = init_capping_rate_row, PA = init_ppr_eq_row, WA = init_wh2_eq_row, E = init_ends_eq_row))

# Set up plot.
plot_hand = figure(tools = "crosshair, pan, reset, save, wheel_zoom", x_axis_label = "Capping rate constant (/s)", y_axis_label = "Steady state")
plot_hand.line('x', 'PA', source = data_source, line_width = 3, line_color = 'Blue', legend = 'PPR')
plot_hand.line('x', 'WA', source = data_source, line_width = 3, line_color = 'Red', legend = 'WH2')
plot_hand.line('x', 'E', source = data_source, line_width = 3, line_color = 'Black', legend = 'Ends')

# Set up widgets.
k_load_slider = Slider(title = 'Load', value = 1.0, start = 1e-3, end = 1.0, step = 0.1)
k_transfer_slider = Slider(title = 'Transfer', value = 0.1, start = 1e-3, end = 1.0, step = 0.1)
k_elongate_slider = Slider(title = 'Elongate', value = 0.5, start = 1e-3, end = 1.0, step = 0.1)
k_nucleate_slider = Slider(title = 'Nucleate', value = 0.05, start = 1e-3, end = 1.0, step = 0.1)

# Set up callbacks.
def update_steady_state(attrname, old, new):
    # Get slider values.
    new_load_const = k_load_slider.value
    new_transfer_const = k_transfer_slider.value
    new_elongate_const = k_elongate_slider.value
    new_nucleate_const = k_nucleate_slider.value

    # Generate new curves.
    new_capping_rate_row = linspace(1e-3, new_nucleate_const, 100)
    new_ppr_eq_row, new_wh2_eq_row, new_ends_eq_row = steady_state(load_const = new_load_const, transfer_const = new_transfer_const, elongation_const = new_elongate_const, nucleation_const = new_nucleate_const)
    data_source.data = dict(x = new_capping_rate_row, PA = new_ppr_eq_row, WA = new_wh2_eq_row, E = new_ends_eq_row)

for i_slider in [k_load_slider, k_transfer_slider, k_elongate_slider, k_nucleate_slider]:
    i_slider.on_change('value', update_steady_state)

# Set up layouts and add to document.
inputs = column(k_load_slider, k_transfer_slider, k_elongate_slider, k_nucleate_slider)
curdoc().add_root(row(inputs, plot_hand))
curdoc().title = "Steady state"
