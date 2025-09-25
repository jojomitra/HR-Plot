import streamlit as st
import pandas as ps
import numpy as np
import random
import matplotlib.pyplot as mlp
from matplotlib.lines import Line2D
from io import BytesIO
import time

# constants used in the program
sigma = 5.670374419e-8
R_sun = 6.957e8
L_sun = 3.828e26


# function to get sorted unique ages in Gyr
@st.cache_data
def get_available_ages(age_data):
    ages = ps.to_numeric(age_data['Age'], errors='coerce').dropna().unique()

    return sorted([round(age / 1e9, 4) for age in ages])


# function to compute the coordinates of stars (log(Teff), Log(L/Lsun))
@st.cache_data
def compute_star_coordinates(Teff, Radius):
    R_m = Radius * R_sun
    L = 4 * np.pi * R_m ** 2 * sigma * Teff ** 4

    return np.log10(Teff), np.log10(L / L_sun)


# function to get unique color for plotting
def get_plot_color():
    while True:
        r, g, b = random.randint(0, 255) / 255, random.randint(0, 255) / 255, random.randint(0, 255) / 255

        color = r, g, b
        brightness = 0.299 * r + 0.587 * g + 0.114 * b

        if (color not in used_colors) and brightness < 0.45:
            used_colors.add(color)

            return color


# function to plot evolutionary tracks
def plot_evol_tracks(fig_axis, z_value, mass_groups_data_list, evol_tracks_color, mass_evol_tracks_list, legend_handles, legend_labels):
    main_seq_line_x = []
    main_seq_line_y = []

    for mass, m_frame in mass_groups_data_list:
        x_logteff_mass = m_frame['Log Teff']
        y_logl_mass = m_frame['Log L']

        evol_track, = fig_axis.plot(x_logteff_mass, y_logl_mass, color=evol_tracks_color, linewidth=0.8, zorder=2)

        main_seq_line_x.append(evol_track.get_xdata()[0])
        main_seq_line_y.append(evol_track.get_ydata()[0])

        mass_evol_tracks_list.append((mass, evol_track))

    # plot main sequence line
    fig_axis.plot(main_seq_line_x, main_seq_line_y, color='darkgrey', linewidth=2.0, zorder=2)

    evol_track_line = Line2D([0], [0], color=evol_tracks_color, linestyle='-', linewidth=1.5)
    legend_handles.append(evol_track_line)
    legend_labels.append(f'Evolutionary Tracks (Z={z_value})')


# function to plot isochrones for specific ages
def plot_isochrones(fig_axis, z_value, age_groups_data_list, legend_handles, legend_labels):
    for age_group_data in age_groups_data_list:
        age = age_group_data.iloc[0]["Age"]

        x_logteff_age = age_group_data['LogT']
        y_logl_age = age_group_data['LogL']

        isochrone_color = st.session_state[f'z{z_value}_a{age / 1e9}']
        fig_axis.scatter(x_logteff_age, y_logl_age, color=isochrone_color, s=10, marker='o', zorder=3)

        isochrone_line = Line2D([0], [0], color=isochrone_color, linestyle=':', linewidth=3.5)
        legend_handles.append(isochrone_line)
        legend_labels.append(f'{np.round(age / 1e9, 4)} Gyr')


# function to plot stars
def plot_stars(fig_axis, name_list, size_list, color_list, x_list, y_list, legend_handles, legend_labels):    
    for star_x, star_y, star_name, star_size, star_color in zip(x_list, y_list, name_list, size_list, color_list):
        fig_axis.scatter(star_x, star_y, color=star_color, s=star_size, marker='*', zorder=3)

        star = Line2D([0], [0], linestyle='', marker='*', color=star_color, markersize=15)
        legend_handles.append(star)
        legend_labels.append(f'Component. {star_name}')


# function to plot HR diagram with evolutionary tracks and isochrones of every selected Z, as well as binary stars
def plot_evol_tracks_isoc_hr(z_values_list, evol_tracks_data_dict, isochrones_data_dict, star_id, star_name_list, star_size_list, star_color_list, star_x_list, star_y_list):
    mlp.rcParams['font.family'] = 'Times New Roman'
    mlp.rcParams['mathtext.fontset'] = 'custom'
    mlp.rcParams['mathtext.rm'] = 'Times New Roman'
    mlp.rcParams['mathtext.it'] = 'Times New Roman:italic'
    mlp.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    fig, ax = mlp.subplots(figsize=(8, 8))

    evol_tracks_cols_list = ['slategray', 'mediumpurple', 'deepskyblue', 'yellowgreen']
    used_colors.update(evol_tracks_cols_list)

    mass_et_list = []
    leg_handles = []
    leg_labels = []

    for i, z_val in enumerate(z_values_list):
        # plot evolutionary tracks for each selected z
        mass_groups = evol_tracks_data_dict[z_val]
        plot_evol_tracks(ax, z_val, mass_groups, evol_tracks_cols_list[i], mass_et_list, leg_handles, leg_labels)

        # plot isochrones for chosen ages of each selected z
        if z_val in isochrones_data_dict:
            age_groups = isochrones_data_dict[z_val]
            plot_isochrones(ax, z_val, age_groups, leg_handles, leg_labels)

    # plot stars
    plot_stars(ax, star_name_list, star_size_list, star_color_list, star_x_list, star_y_list, leg_handles, leg_labels)

    ax.set_xlabel(r'log$T_{\text{eff}}$', fontsize=20, labelpad=10)
    ax.set_ylabel(r'log($L/L_{\odot}$)', fontsize=20, labelpad=5)
    ax.tick_params(labelsize=15)
    ax.invert_xaxis()
    ax.legend(handles=leg_handles,
              labels=leg_labels,
              title=f'{star_id}',
              title_fontsize=15, fontsize=13, edgecolor='grey', loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.grid(visible=True, color='lightgrey', linewidth=0.5, zorder=1)

    return fig, ax, mass_et_list


# function to label evolutionary tracks with their corresponding mass values
def label_evol_tracks_with_mass(fig_axis, mass_evol_tracks_list):
    x_lim = fig_axis.get_xlim()
    y_lim = fig_axis.get_ylim()

    x_axis_inc = np.abs(fig_axis.get_xticks()[0] - fig_axis.get_xticks()[1])
    y_axis_inc = np.abs(fig_axis.get_yticks()[0] - fig_axis.get_yticks()[1])

    for mass, evol_track in mass_evol_tracks_list:
        mass_text_x = evol_track.get_xdata()[0] + (0.01 * x_axis_inc)
        mass_text_y = evol_track.get_ydata()[0] - (0.3 * y_axis_inc)

        # only display mass text labels that appear inside the figure
        if (min(x_lim) < mass_text_x < max(x_lim)) and (min(y_lim) < mass_text_y < max(y_lim)):
            fig_axis.text(mass_text_x, mass_text_y, f'{mass}', fontsize=12, weight='bold', rotation=60, rotation_mode='anchor')


if 'session_start' not in st.session_state:
    st.session_state['session_start'] = time.time()

session_run_time = time.time() - st.session_state['session_start']

if session_run_time > 900:
    st.session_state.clear()

st.set_page_config(layout='centered')

st.title('Plotting HR Diagram - Part 4 of Al-Wardat Method for Analyzing Stellar Systems and Variable Stars')
st.markdown("""You can use this code to plot stars on the Hertzsprung-Russell (HR) diagram, depicting evolutionary tracks based on the theoretical models of Gerardi (2000a, b). 

This code was developed under the supervision of "[Prof. Mashhoor Al-Wardat](https://www.sharjah.ac.ae/en/Academics/Faculty-And-Staff/Mashhoor-Ahmad-Salameh)" (malwardat@sharjah.ac.ae) as part of Al-Wardat’s Method for Analyzing Stellar systems and Variable Stars. 

The Python code was created by :rainbow[Ms. Kaivisna Kandan] (U21103479@sharjah.ac.ae), :rainbow[Ms. Fidha Sirajudheen] (U23102657@sharjah.ac.ae), and :rainbow[Ms. Mariam Ismail] (U22101102@sharjah.ac.ae), with assistance from :rainbow[Mr. Hassan Haboubi] (U23103604@sharjah.ac.ae).

Enter the Teff and radius of each star, then choose the appropriate metallicity and age to fit the stellar system. The output will be a visualization of luminosity (log(L/L_sun)) versus effective temperature (logT_eff).
""")
st.subheader('Select Metallicity (Z) and Isochrone Age')

# z value selection
z_range = [0.0004, 0.008, 0.019, 0.03]
z_values = st.multiselect('Metallicities (Z):', z_range)

et_data_dict = {}
isoc_data_dict = {}

used_colors = set()

for z in z_values:
    # load corresponding z value file for evolutionary tracks and isochrones
    evol_tracks_file = f"cleaned_Z={z}.csv"
    isochrones_file = f"cleaned_Z={z}_age.csv"

    evol_tracks_df = ps.read_csv(evol_tracks_file)
    isochrones_df = ps.read_csv(isochrones_file)

    # isochrone ages for each z value selection
    available_ages = get_available_ages(isochrones_df)
    age_values = st.multiselect(f'Isochrone ages (Gyr) for Z={z}:', available_ages)

    # group evolutionary tracks data for each z value based on initial mass
    et_mass_groups = evol_tracks_df.groupby('Initial Mass')
    et_data_dict[z] = et_mass_groups

    # store data of every selected isochrone age for each z value
    isoc_data_dict[z] = []

    for age in age_values:
        isoc_age_data = isochrones_df.loc[isochrones_df['Age'] == (age * 1e9)]
        isoc_data_dict[z].append(isoc_age_data)

        if f'z{z}_a{age}' not in st.session_state:
            st.session_state[f'z{z}_a{age}'] = get_plot_color()

# star parameters
st.subheader('Enter Star Parameters')

star_identifier = st.text_input("Enter the star identifier:")

num_stars = st.number_input('Enter the number of star components:', min_value=0, max_value=5, step=1)

x_list = []
y_list = []
star_component_name_list = []
star_component_size_list = []
star_component_color_list = []

star_colors = ['#F90004', '#2800F9', '#026D0F', '#DC570A', '#DC0ACB']

for i in range(num_stars):
    T_value = R_value = None

    # input for star parameters
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        star_component_name = st.text_input(f"Component{i+1} name:")        
    with col2:
        T = st.text_input(f'Temperature T{i+1} (K):')
        if T:
            try:
                T_value = float(T)
            except ValueError:
                st.error(f'T{i+1} must be a number')
    with col3:
        R = st.text_input(f'Radius R{i+1} (R\u2299):')
        if R:
            try:
                R_value = float(R)
            except ValueError:
                st.error(f'R{i+1} must be a number')
    with col4:
        star_component_size = st.number_input(f"Component{i+1} size:", min_value=100, value=300, step=10)
    with col5:
        star_component_color = st.color_picker("Choose a color", star_colors[i])

    if star_component_name and star_component_size and star_component_color:
        star_component_name_list.append(star_component_name)
        star_component_size_list.append(star_component_size)
        star_component_color_list.append(star_component_color)

    # compute star coordinates (log(Teff), Log(L/Lsun))
    if T_value and R_value:
        x, y = compute_star_coordinates(T_value, R_value)

        x_list.append(x)
        y_list.append(y)

# display HR diagram with evolutionary tracks and isochrones
hr_fig = hr_ax = mass_et_list = None

if z_values and et_data_dict and star_identifier and 0 < num_stars == len(star_component_name_list) == len(star_component_size_list) == len(star_component_color_list) == len(x_list) == len(y_list):
    hr_fig, hr_ax, mass_et_list = plot_evol_tracks_isoc_hr(z_values, et_data_dict, isoc_data_dict, star_identifier, star_component_name_list, star_component_size_list, star_component_color_list, x_list, y_list)
    st.pyplot(hr_fig)

    # save full plot
    buf_fullplot = BytesIO()
    hr_fig.savefig(buf_fullplot, format="png", bbox_inches="tight")
    buf_fullplot.seek(0)

    st.download_button("Download Full Plot", data=buf_fullplot, mime="image/png")

# input for HR plot's display range
st.subheader('Select Display Range for HR Plot')

max_logteff_range_value = min_logteff_range_value = max_logl_range_value = min_logl_range_value = None

col1_range, col2_range = st.columns(2)
with col1_range:
    min_logteff_range = st.text_input('Minimum value of log(Teff):')
    if min_logteff_range:
        try:
            min_logteff_range_value = float(min_logteff_range)
        except ValueError:
            st.error('Minimum log(Teff) must be a number')

    min_logl_range = st.text_input('Minimum value of log(L/L\u2299):')
    if min_logl_range:
        try:
            min_logl_range_value = float(min_logl_range)
        except ValueError:
            st.error('Minimum log(L/L\u2299) must be a number')
with col2_range:
    max_logteff_range = st.text_input('Maximum value of log(Teff):')
    if max_logteff_range:
        try:
            max_logteff_range_value = float(max_logteff_range)
        except ValueError:
            st.error('Maximum log(Teff) must be a number')

    max_logl_range = st.text_input('Maximum value of log(L/L\u2299):')
    if max_logl_range:
        try:
            max_logl_range_value = float(max_logl_range)
        except ValueError:
            st.error('Maximum log(L/L\u2299) must be a number')

# display HR plot within the selected display range
selected_x_range = bool(max_logteff_range_value and min_logteff_range_value)
selected_y_range = bool(max_logl_range_value and min_logl_range_value)

if (selected_x_range or selected_y_range) and hr_fig and hr_ax and mass_et_list:
    if selected_x_range:
        hr_ax.set_xlim([max_logteff_range_value, min_logteff_range_value])

    if selected_y_range:
        hr_ax.set_ylim([min_logl_range_value, max_logl_range_value])

    # label evolutionary tracks with mass values
    label_evol_tracks_with_mass(hr_ax, mass_et_list)

    st.pyplot(hr_fig)

    # save the ranged plot
    buf_rangeplot = BytesIO()
    hr_fig.savefig(buf_rangeplot, format="png", bbox_inches="tight")
    buf_rangeplot.seek(0)

    st.download_button("Download Ranged Plot", data=buf_rangeplot, mime="image/png")
