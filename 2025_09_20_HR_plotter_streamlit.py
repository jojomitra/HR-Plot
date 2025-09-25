import streamlit as st
import pandas as ps
import numpy as np
import random
import matplotlib.pyplot as mlp
from matplotlib.lines import Line2D
from io import BytesIO
import time
import os

# Try to import a fast KD-tree implementation (scipy preferred, sklearn fallback)
KDTreeImpl = None
KD_IMPL = None
try:
    from scipy.spatial import cKDTree as KDTreeImpl
    KD_IMPL = "scipy"
except Exception:
    try:
        from sklearn.neighbors import KDTree as KDTreeImpl
        KD_IMPL = "sklearn"
    except Exception:
        KDTreeImpl = None
        KD_IMPL = None

# constants
sigma = 5.670374419e-8
R_sun = 6.957e8
L_sun = 3.828e26

# ----------------------------
# Helper functions & plotting
# ----------------------------

@st.cache_data
def get_available_ages(age_data):
    """Return sorted ages in Gyr from an isochrone dataframe."""
    if age_data is None or age_data.empty:
        return []
    ages = ps.to_numeric(age_data['Age'], errors='coerce').dropna().unique()
    return sorted([round(age / 1e9, 4) for age in ages])

@st.cache_data
def compute_star_coordinates(Teff, Radius):
    """Return log10(Teff) and log10(L/L_sun)."""
    R_m = Radius * R_sun
    L = 4 * np.pi * R_m ** 2 * sigma * Teff ** 4
    return np.log10(Teff), np.log10(L / L_sun)

def get_plot_color():
    """Return non-too-bright random RGB tuple; ensures uniqueness via used_colors set."""
    while True:
        r, g, b = random.randint(0, 255) / 255, random.randint(0, 255) / 255, random.randint(0, 255) / 255
        color = (r, g, b)
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        if (color not in used_colors) and brightness < 0.45:
            used_colors.add(color)
            return color

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

    fig_axis.plot(main_seq_line_x, main_seq_line_y, color='darkgrey', linewidth=2.0, zorder=2)
    evol_track_line = Line2D([0], [0], color=evol_tracks_color, linestyle='-', linewidth=1.5)
    legend_handles.append(evol_track_line)
    legend_labels.append(f'Evolutionary Tracks (Z={z_value})')

def plot_isochrones(fig_axis, z_value, age_groups_data_list, legend_handles, legend_labels):
    for age_group_data in age_groups_data_list:
        if age_group_data is None or age_group_data.empty:
            continue
        age = age_group_data.iloc[0]["Age"]
        x_logteff_age = age_group_data['LogT']
        y_logl_age = age_group_data['LogL']
        isochrone_color = st.session_state.get(f'z{z_value}_a{age / 1e9}', get_plot_color())
        fig_axis.scatter(x_logteff_age, y_logl_age, color=isochrone_color, s=10, marker='o', zorder=3)
        isochrone_line = Line2D([0], [0], color=isochrone_color, linestyle=':', linewidth=3.5)
        legend_handles.append(isochrone_line)
        legend_labels.append(f'{np.round(age / 1e9, 4)} Gyr')

def plot_stars(fig_axis, name_list, size_list, color_list, x_list, y_list, legend_handles, legend_labels):
    for star_x, star_y, star_name, star_size, star_color in zip(x_list, y_list, name_list, size_list, color_list):
        fig_axis.scatter(star_x, star_y, color=star_color, s=star_size, marker='*', zorder=3)
        star = Line2D([0], [0], linestyle='', marker='*', color=star_color, markersize=15)
        legend_handles.append(star)
        legend_labels.append(f'Component. {star_name}')

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
        mass_groups = evol_tracks_data_dict.get(z_val, [])
        plot_evol_tracks(ax, z_val, mass_groups, evol_tracks_cols_list[i % len(evol_tracks_cols_list)], mass_et_list, leg_handles, leg_labels)
        if z_val in isochrones_data_dict:
            age_groups = isochrones_data_dict[z_val]
            plot_isochrones(ax, z_val, age_groups, leg_handles, leg_labels)

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

def label_evol_tracks_with_mass(fig_axis, mass_evol_tracks_list):
    x_lim = fig_axis.get_xlim()
    y_lim = fig_axis.get_ylim()
    x_axis_inc = np.abs(fig_axis.get_xticks()[0] - fig_axis.get_xticks()[1])
    y_axis_inc = np.abs(fig_axis.get_yticks()[0] - fig_axis.get_yticks()[1])

    for mass, evol_track in mass_evol_tracks_list:
        mass_text_x = evol_track.get_xdata()[0] + (0.01 * x_axis_inc)
        mass_text_y = evol_track.get_ydata()[0] - (0.3 * y_axis_inc)
        if (min(x_lim) < mass_text_x < max(x_lim)) and (min(y_lim) < mass_text_y < max(y_lim)):
            fig_axis.text(mass_text_x, mass_text_y, f'{mass}', fontsize=12, weight='bold', rotation=60, rotation_mode='anchor')

# ----------------------------
# Safe cached loader for isochrone age points
# ----------------------------
@st.cache_data
def load_iso_age_points(filename):
    """
    Returns a dict {age_years: points_array Nx2} where points_array is np.ndarray of (LogT, LogL).
    Only scalar/array data is returned (no KD-tree objects), so caching is safe.
    """
    if not os.path.exists(filename):
        return {}
    try:
        df = ps.read_csv(filename)
    except Exception:
        return {}
    out = {}
    ages = sorted(df['Age'].dropna().unique())
    for age in ages:
        df_age = df[df['Age'] == age]
        pts = np.vstack([df_age['LogT'].values, df_age['LogL'].values]).T
        if pts.size == 0:
            continue
        out[int(age)] = pts
    return out

def kd_mean_distance_to_points_using_kd(pts, star_coords_arr):
    """
    Given pts (Nx2) and star_coords_arr (Mx2 numpy array), compute mean of minimal Euclidean distances
    from each star to the pts using KD-tree only. If KD implementation missing, return np.inf.
    """
    if KDTreeImpl is None:
        # explicit behavior: no KD -> do not compute, return infinite score so Z is not suggested
        return np.inf

    if pts is None or getattr(pts, "size", 0) == 0:
        return np.inf
    if star_coords_arr is None:
        return np.inf
    star_arr = np.asarray(star_coords_arr)
    if star_arr.size == 0:
        return np.inf

    # Build KD tree and query
    tree = KDTreeImpl(pts)
    res = tree.query(star_arr, k=1)
    # res may be tuple (dists, idx) or array; handle both
    if isinstance(res, tuple) and len(res) >= 1:
        dists = np.asarray(res[0]).reshape(-1)
    else:
        dists = np.asarray(res).reshape(-1)
    return float(np.mean(dists))

def rank_z_by_match_require_kd(z_values_all, basepath, star_coords_tuple):
    """
    For each Z, load isochrone age points (cached arrays) and compute best age score using KD only.
    Returns list [(z, best_age_years_or_None, best_score), ...] sorted by score ascending.
    If KD implementation missing, returns all np.inf scores and emits an error later in UI.
    """
    results = []
    if not star_coords_tuple:
        return [(z, None, np.inf) for z in z_values_all]
    star_coords_arr = np.array([list(t) for t in star_coords_tuple])  # shape (M,2)

    for z in z_values_all:
        fname = os.path.join(basepath, f"cleaned_Z={z}_age.csv")
        if not os.path.exists(fname):
            fname = f"cleaned_Z={z}_age.csv"
        if not os.path.exists(fname):
            results.append((z, None, np.inf))
            continue

        age_pts = load_iso_age_points(fname)  # cached arrays
        if not age_pts:
            results.append((z, None, np.inf))
            continue

        best_age = None
        best_score = np.inf
        # iterate ages and demand KD-based distances
        for age, pts in age_pts.items():
            s = kd_mean_distance_to_points_using_kd(pts, star_coords_arr)
            if s < best_score:
                best_score = s
                best_age = age
        results.append((z, best_age, best_score))
    # sort such that finite scores come first
    results.sort(key=lambda t: (t[2] == np.inf, t[2]))
    return results

# ----------------------------
# App UI flow
# ----------------------------
if 'session_start' not in st.session_state:
    st.session_state['session_start'] = time.time()

session_run_time = time.time() - st.session_state['session_start']
# clear after 15 minutes to avoid stale state (same behavior as before)
if session_run_time > 900:
    st.session_state.clear()

st.set_page_config(layout='centered')

st.title('HR Diagram Plotter — KD-only suggestions, auto-applied')
st.markdown("""
This version **requires** a KD-tree implementation (`scipy` or `scikit-learn`).  
It automatically applies the top metallicity (Z) and top ages after you press **Apply star inputs**.  
If you prefer a different selection, just change the multiselects manually.
""")

# 1) Collect star inputs inside a form
with st.form("star_input_form", clear_on_submit=False):
    st.subheader("1) Star inputs (press Apply star inputs when done)")
    star_identifier = st.text_input("Star identifier:", value=st.session_state.get("star_identifier", ""))
    num_stars = st.number_input('Number of star components:', min_value=0, max_value=5, value=int(st.session_state.get("num_stars", 0)))
    temp_names = []
    temp_Ts = []
    temp_Rs = []
    temp_sizes = []
    temp_colors = []
    star_colors_defaults = ['#F90004', '#2800F9', '#026D0F', '#DC570A', '#DC0ACB']

    for i in range(int(num_stars)):
        cols = st.columns([2, 2, 2, 1, 2])
        with cols[0]:
            name = st.text_input(f"Component {i+1} name:", value=(st.session_state.get(f'comp{i+1}_name') or ""))
        with cols[1]:
            tempT = st.text_input(f"T{i+1} (K):", value=(st.session_state.get(f'comp{i+1}_T') or ""))
        with cols[2]:
            tempR = st.text_input(f"R{i+1} (R\u2299):", value=(st.session_state.get(f'comp{i+1}_R') or ""))
        with cols[3]:
            size = st.number_input(f"size{i+1} (plot):", min_value=50, value=int(st.session_state.get(f'comp{i+1}_size') or 300), step=10, key=f"size_{i}")
        with cols[4]:
            color_default = st.session_state.get(f'comp{i+1}_color', star_colors_defaults[i])
            color = st.color_picker(f"color{i+1}:", value=color_default, key=f"color_{i}")

        temp_names.append(name)
        temp_Ts.append(tempT)
        temp_Rs.append(tempR)
        temp_sizes.append(size)
        temp_colors.append(color)

    apply_submit = st.form_submit_button("Apply star inputs")
    if apply_submit:
        # Validate & compute coordinates
        valid = True
        x_list = []
        y_list = []
        final_names = []
        final_sizes = []
        final_colors = []
        for i in range(int(num_stars)):
            name = temp_names[i].strip()
            Tstr = temp_Ts[i].strip()
            Rstr = temp_Rs[i].strip()
            size = int(temp_sizes[i])
            color = temp_colors[i]

            if not name:
                st.error(f"Component {i+1} name is required.")
                valid = False
                break
            try:
                Tval = float(Tstr)
                Rval = float(Rstr)
            except Exception:
                st.error(f"Component {i+1} T and R must be valid numbers.")
                valid = False
                break

            x, y = compute_star_coordinates(Tval, Rval)
            x_list.append(x)
            y_list.append(y)
            final_names.append(name)
            final_sizes.append(size)
            final_colors.append(color)

            # persist raw inputs
            st.session_state[f'comp{i+1}_name'] = name
            st.session_state[f'comp{i+1}_T'] = Tstr
            st.session_state[f'comp{i+1}_R'] = Rstr
            st.session_state[f'comp{i+1}_size'] = size
            st.session_state[f'comp{i+1}_color'] = color

        if valid:
            # store computed star coords + metadata
            st.session_state['star_identifier'] = star_identifier
            st.session_state['num_stars'] = int(num_stars)
            st.session_state['star_names'] = final_names
            st.session_state['star_sizes'] = final_sizes
            st.session_state['star_colors'] = final_colors
            st.session_state['star_x'] = x_list
            st.session_state['star_y'] = y_list
            st.success("Star inputs applied — suggestions computed below (KD-only).")

# initialize used_colors set (for color assignment)
used_colors = set()

# candidate metallicities
z_range = [0.0004, 0.008, 0.019, 0.03]

# retrieve star coords from session_state
star_x = st.session_state.get('star_x', [])
star_y = st.session_state.get('star_y', [])
star_names = st.session_state.get('star_names', [])
star_sizes = st.session_state.get('star_sizes', [])
star_colors = st.session_state.get('star_colors', [])
star_identifier = st.session_state.get('star_identifier', "")

# If KD implementation not present, inform user and skip auto-suggestions
if KDTreeImpl is None:
    st.error("KD-tree implementation not found. Please install `scipy` (recommended) or `scikit-learn`. Without KD the app will not suggest metallicity/age automatically.")
    z_rank_results = []
    suggested_zs = []
else:
    # Compute suggestions automatically if we have star coords
    suggested_zs = []
    z_rank_results = []
    if star_x and star_y:
        star_coords_tuple = tuple((float(x), float(y)) for x, y in zip(star_x, star_y))
        with st.spinner("Ranking metallicities (KD-tree matching)..."):
            z_rank_results = rank_z_by_match_require_kd(tuple(z_range), ".", star_coords_tuple)
        finite_results = [r for r in z_rank_results if r[2] != np.inf]
        if finite_results:
            # auto-apply best Z and compute default top ages for that Z
            best_z = finite_results[0][0]
            suggested_zs = [best_z]
            st.session_state['selected_z_values'] = suggested_zs

            # compute top N ages for best_z using KD
            fname = f"cleaned_Z={best_z}_age.csv"
            if os.path.exists(fname):
                age_pts = load_iso_age_points(fname)
                top_n = 3
                age_scores = []
                star_coords_arr = np.array([list(t) for t in star_coords_tuple])
                for age, pts in age_pts.items():
                    s = kd_mean_distance_to_points_using_kd(pts, star_coords_arr)
                    age_scores.append((age, s))
                age_scores.sort(key=lambda t: t[1])
                top_ages = [round(a / 1e9, 4) for a, sc in age_scores[:top_n] if sc != np.inf]
                st.session_state[f'age_defaults_z{best_z}'] = top_ages

# Display ranking table (if any results)
if z_rank_results:
    st.subheader("Z ranking (best matches first)")
    rows = []
    for z, best_age, score in z_rank_results:
        if score == np.inf:
            rows.append({"Z": z, "Best age (Gyr)": "N/A", "Score": "N/A (file missing or no KD match)"})
        else:
            ag_gyr = round(best_age / 1e9, 4) if best_age is not None else "N/A"
            rows.append({"Z": z, "Best age (Gyr)": ag_gyr, "Score": f"{score:.6e}"})
    st.table(ps.DataFrame(rows))

# Z and ages multiselect UI (defaults auto-applied from session_state if available)
st.subheader("Select metallicities (Z) and isochrone ages")
z_default = st.session_state.get('selected_z_values', suggested_zs)
z_values = st.multiselect('Metallicities (Z):', z_range, default=z_default, key='multiselect_z')

et_data_dict = {}
isoc_data_dict = {}

for z in z_values:
    evol_tracks_file = f"cleaned_Z={z}.csv"
    isochrones_file = f"cleaned_Z={z}_age.csv"

    if os.path.exists(evol_tracks_file):
        evol_tracks_df = ps.read_csv(evol_tracks_file)
        et_data_dict[z] = evol_tracks_df.groupby('Initial Mass')
    else:
        et_data_dict[z] = []
        st.warning(f"Evol tracks file missing for Z={z}: {evol_tracks_file}")

    if os.path.exists(isochrones_file):
        isochrones_df = ps.read_csv(isochrones_file)
    else:
        isochrones_df = ps.DataFrame()
        st.warning(f"Isochrone file missing for Z={z}: {isochrones_file}")

    isoc_data_dict[z] = []
    available_ages = get_available_ages(isochrones_df)
    # Default ages come from session_state age_defaults_z{z} (auto-suggested) or previous selection
    age_default = st.session_state.get(f'age_select_z{z}') or st.session_state.get(f'age_defaults_z{z}') or []
    age_values = st.multiselect(f'Isochrone ages (Gyr) for Z={z}:', available_ages, default=age_default, key=f'age_select_z{z}')

    for age in age_values:
        if not isochrones_df.empty:
            df_age = isochrones_df.loc[isochrones_df['Age'] == (age * 1e9)]
            isoc_data_dict[z].append(df_age)
            if f'z{z}_a{age}' not in st.session_state:
                st.session_state[f'z{z}_a{age}'] = get_plot_color()

# Plotting (unchanged)
hr_fig = hr_ax = mass_et_list = None
if z_values and et_data_dict and star_identifier and 0 < len(star_names) == len(star_sizes) == len(star_colors) == len(star_x) == len(star_y):
    hr_fig, hr_ax, mass_et_list = plot_evol_tracks_isoc_hr(z_values, et_data_dict, isoc_data_dict, star_identifier, star_names, star_sizes, star_colors, star_x, star_y)
    st.pyplot(hr_fig)

    buf_fullplot = BytesIO()
    hr_fig.savefig(buf_fullplot, format="png", bbox_inches="tight")
    buf_fullplot.seek(0)
    st.download_button("Download Full Plot", data=buf_fullplot, mime="image/png")

# Display range controls (unchanged)
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

selected_x_range = bool(max_logteff_range_value and min_logteff_range_value)
selected_y_range = bool(max_logl_range_value and min_logl_range_value)

if (selected_x_range or selected_y_range) and hr_fig and hr_ax and mass_et_list:
    if selected_x_range:
        hr_ax.set_xlim([max_logteff_range_value, min_logteff_range_value])
    if selected_y_range:
        hr_ax.set_ylim([min_logl_range_value, max_logl_range_value])

    label_evol_tracks_with_mass(hr_ax, mass_et_list)
    st.pyplot(hr_fig)

    buf_rangeplot = BytesIO()
    hr_fig.savefig(buf_rangeplot, format="png", bbox_inches="tight")
    buf_rangeplot.seek(0)
    st.download_button("Download Ranged Plot", data=buf_rangeplot, mime="image/png")
