import streamlit as st
import pandas as ps
import numpy as np
import random
import matplotlib.pyplot as mlp
from matplotlib.lines import Line2D
from io import BytesIO
import time
import os

# Try to import a fast KD-tree implementation
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

# constants used in the program
sigma = 5.670374419e-8
R_sun = 6.957e8
L_sun = 3.828e26

# -----------------------------------------------------------------------------
# Utility & plotting functions (mostly unchanged)
# -----------------------------------------------------------------------------

# function to get sorted unique ages in Gyr
@st.cache_data
def get_available_ages(age_data):
    if age_data is None or age_data.empty:
        return []
    ages = ps.to_numeric(age_data['Age'], errors='coerce').dropna().unique()
    return sorted([round(age / 1e9, 4) for age in ages])

# function to compute the coordinates of stars (log(Teff), Log(L/Lsun))
@st.cache_data
def compute_star_coordinates(Teff, Radius):
    R_m = Radius * R_sun
    L = 4 * np.pi * R_m ** 2 * sigma * Teff ** 4
    return np.log10(Teff), np.log10(L / L_sun)

# function to get unique color for plotting (ensures not-too-bright)
def get_plot_color():
    while True:
        r, g, b = random.randint(0, 255) / 255, random.randint(0, 255) / 255, random.randint(0, 255) / 255
        color = (r, g, b)
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
        mass_groups = evol_tracks_data_dict.get(z_val, [])
        plot_evol_tracks(ax, z_val, mass_groups, evol_tracks_cols_list[i % len(evol_tracks_cols_list)], mass_et_list, leg_handles, leg_labels)

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

# label evolutionary tracks with their corresponding mass values
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

# -----------------------------------------------------------------------------
# KD-tree matching utilities (cached)
# -----------------------------------------------------------------------------

@st.cache_data
def build_age_kdtrees(isochrones_df):
    """
    Given an isochrones dataframe for one Z, builds KD-trees for each age.
    Returns dict: {age_years: (tree, points_array Nx2)}.
    Points are in original (LogT, LogL) coordinates.
    """
    if isochrones_df is None or isochrones_df.empty:
        return {}
    if KDTreeImpl is None:
        # No KD implementation available
        return {}

    trees = {}
    ages = sorted(isochrones_df['Age'].dropna().unique())
    for age in ages:
        df_age = isochrones_df[isochrones_df['Age'] == age]
        pts = np.vstack([df_age['LogT'].values, df_age['LogL'].values]).T
        if pts.size == 0:
            continue
        # build KD-tree
        try:
            tree = KDTreeImpl(pts)
        except Exception:
            # some implementations expect 2D numeric arrays; if it fails, skip
            continue
        trees[age] = (tree, pts)
    return trees

@st.cache_data
def kd_score_age(tree_pts_tuple, star_coords):
    """
    Given (tree, pts) and list of star_coords [(x,y), ...], return mean minimal Euclidean distance.
    """
    if not tree_pts_tuple or not star_coords:
        return np.inf
    tree, pts = tree_pts_tuple
    # KDTree query: tree.query accepts either (n_samples, n_features) or (single_point,).
    arr = np.array(star_coords)
    try:
        dists, _ = tree.query(arr, k=1)
    except Exception:
        # sklearn.KDTree returns (distances, indices) but perhaps in a different order
        try:
            res = tree.query(arr, k=1)
            dists = res[0]
        except Exception:
            return np.inf
    # flatten to 1D distances
    dists = np.array(dists).reshape(-1)
    return float(np.mean(dists))

@st.cache_data
def rank_z_by_match(z_values_all, basepath, star_coords):
    """
    For each Z in z_values_all tries to load the isochrone file and compute best matching age (lowest KD mean distance).
    Returns a list sorted by best_score ascending: [(z, best_age_years_or_none, best_score), ...]
    """
    results = []
    for z in z_values_all:
        fname = os.path.join(basepath, f"cleaned_Z={z}_age.csv")
        if not os.path.exists(fname):
            fname = f"cleaned_Z={z}_age.csv"  # fallback to working dir
        if not os.path.exists(fname):
            results.append((z, None, np.inf))
            continue
        try:
            isoc_df = ps.read_csv(fname)
        except Exception:
            results.append((z, None, np.inf))
            continue

        trees = build_age_kdtrees(isoc_df)
        if not trees:
            results.append((z, None, np.inf))
            continue

        best_age = None
        best_score = np.inf
        for age, tree_tuple in trees.items():
            s = kd_score_age(tree_tuple, star_coords)
            if s < best_score:
                best_score = s
                best_age = age
        results.append((z, best_age, best_score))
    # sort with finite scores first by score
    results.sort(key=lambda t: (t[2] == np.inf, t[2]))
    return results

# -----------------------------------------------------------------------------
# App UI & flow
# -----------------------------------------------------------------------------

if 'session_start' not in st.session_state:
    st.session_state['session_start'] = time.time()

session_run_time = time.time() - st.session_state['session_start']
# clear session after 15 minutes (like your original)
if session_run_time > 900:
    st.session_state.clear()

st.set_page_config(layout='centered')

st.title('HR Diagram Plotter — KD-tree suggestions + apply-button UI')
st.markdown("""
Enter star effective temperature (Teff in K) and radius (R in R\N{SUN}) for each component.
Press **Apply star inputs** to compute suggested metallicity (Z) and isochrone ages.
You can then press **Apply top suggestions** to auto-fill the Z and Age selectors, and then generate the plot.
""")

# ---------------------------------------------------------------------
# 1) Collect star inputs inside a form so streamlit doesn't re-run on each field change
# ---------------------------------------------------------------------
with st.form("star_input_form", clear_on_submit=False):
    st.subheader("1) Star input form")
    star_identifier = st.text_input("Star identifier:", value=st.session_state.get("star_identifier", ""))
    num_stars = st.number_input('Number of star components:', min_value=0, max_value=5, value=int(st.session_state.get("num_stars", 0)))
    # containers to store raw inputs temporarily
    temp_names = []
    temp_Ts = []
    temp_Rs = []
    temp_sizes = []
    temp_colors = []

    star_colors_defaults = ['#F90004', '#2800F9', '#026D0F', '#DC570A', '#DC0ACB']

    for i in range(num_stars):
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
        # validate and store in session_state
        valid = True
        x_list = []
        y_list = []
        final_names = []
        final_sizes = []
        final_colors = []
        for i in range(num_stars):
            name = temp_names[i].strip()
            Tstr = temp_Ts[i].strip()
            Rstr = temp_Rs[i].strip()
            size = int(temp_sizes[i])
            color = temp_colors[i]

            # basic validation
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

            # compute coords
            x, y = compute_star_coordinates(Tval, Rval)
            x_list.append(x)
            y_list.append(y)
            final_names.append(name)
            final_sizes.append(size)
            final_colors.append(color)

            # store raw inputs so they persist in fields
            st.session_state[f'comp{i+1}_name'] = name
            st.session_state[f'comp{i+1}_T'] = Tstr
            st.session_state[f'comp{i+1}_R'] = Rstr
            st.session_state[f'comp{i+1}_size'] = size
            st.session_state[f'comp{i+1}_color'] = color

        if valid:
            st.session_state['star_identifier'] = star_identifier
            st.session_state['num_stars'] = num_stars
            st.session_state['star_names'] = final_names
            st.session_state['star_sizes'] = final_sizes
            st.session_state['star_colors'] = final_colors
            st.session_state['star_x'] = x_list
            st.session_state['star_y'] = y_list
            st.success("Star inputs applied — suggestions being prepared below.")

# ---------------------------------------------------------------------
# 2) If star inputs are present in session_state, compute KD-tree matches and suggest Z & ages
# ---------------------------------------------------------------------
used_colors = set()  # reset / initialize used_colors

z_range = [0.0004, 0.008, 0.019, 0.03]  # available metallicities (adjust if you have more)

star_x = st.session_state.get('star_x', [])
star_y = st.session_state.get('star_y', [])
star_names = st.session_state.get('star_names', [])
star_sizes = st.session_state.get('star_sizes', [])
star_colors = st.session_state.get('star_colors', [])
star_identifier = st.session_state.get('star_identifier', "")

suggested_zs = []
z_rank_results = []
if star_x and star_y and (KDTreeImpl is not None):
    star_coords = list(zip(star_x, star_y))
    with st.spinner("Ranking metallicities (fast KD-tree matching)..."):
        z_rank_results = rank_z_by_match(z_range, ".", star_coords)
    # compute suggested_zs from results (take first finite-score result)
    finite_results = [r for r in z_rank_results if r[2] != np.inf]
    if finite_results:
        suggested_zs = [finite_results[0][0]]  # best single suggestion
else:
    if (KDTreeImpl is None) and (star_x and star_y):
        st.error("No KD-tree implementation found. Install scipy or scikit-learn (see requirements).")

# Display Z ranking summary
if z_rank_results:
    st.subheader("Z ranking (best matches first)")
    rows = []
    for z, best_age, score in z_rank_results:
        if score == np.inf:
            rows.append({"Z": z, "Best age (Gyr)": "N/A", "Score": "N/A (file missing)"})
        else:
            ag_gyr = round(best_age / 1e9, 4) if best_age is not None else "N/A"
            rows.append({"Z": z, "Best age (Gyr)": ag_gyr, "Score": f"{score:.6e}"})
    st.table(ps.DataFrame(rows))

# Provide an "Apply top suggestions" button that will prefill selectors and rerun
if suggested_zs:
    if st.button("Apply top suggestions (prefill best Z and top ages)"):
        # store chosen Zs in session state so that multiselect defaults will show them
        st.session_state['selected_z_values'] = suggested_zs
        # For each suggested z, compute top 3 ages and store default selections for their multiselect keys
        star_coords = list(zip(star_x, star_y))
        for z in suggested_zs:
            fname = f"cleaned_Z={z}_age.csv"
            if not os.path.exists(fname):
                continue
            df_isoc = ps.read_csv(fname)
            trees = build_age_kdtrees(df_isoc)
            age_scores = []
            for age, ttuple in trees.items():
                s = kd_score_age(ttuple, star_coords)
                age_scores.append((age, s))
            age_scores.sort(key=lambda t: t[1])
            top_ages = [round(a / 1e9, 4) for a, sc in age_scores[:3] if sc != np.inf]
            st.session_state[f'age_defaults_z{z}'] = top_ages
        # force a rerun so UI updates using new session_state defaults
        st.experimental_rerun()

# ---------------------------------------------------------------------
# 3) Z & age selection UI (multiselects). Prefill defaults from session_state where available.
# ---------------------------------------------------------------------
st.subheader("Select metallicities (Z) and isochrone ages (you can accept suggested values)")

# default for Z selector: either user previously selected, session selected, or suggested_zs
z_default = st.session_state.get('selected_z_values', suggested_zs)
z_values = st.multiselect('Metallicities (Z):', z_range, default=z_default, key='multiselect_z')

# For each selected Z, show the ages multiselect; default from session_state[f'age_defaults_z{z}'] or previously chosen
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
    # defaults: 1) any explicit previously chosen key; 2) prefills from Apply suggestions; 3) empty
    age_default = st.session_state.get(f'age_select_z{z}') or st.session_state.get(f'age_defaults_z{z}') or []
    age_values = st.multiselect(f'Isochrone ages (Gyr) for Z={z}:', available_ages, default=age_default, key=f'age_select_z{z}')

    for age in age_values:
        if not isochrones_df.empty:
            df_age = isochrones_df.loc[isochrones_df['Age'] == (age * 1e9)]
            isoc_data_dict[z].append(df_age)
            # ensure color exists for this age selection
            if f'z{z}_a{age}' not in st.session_state:
                st.session_state[f'z{z}_a{age}'] = get_plot_color()

# ---------------------------------------------------------------------
# 4) Plotting (same logic as before)
# ---------------------------------------------------------------------
hr_fig = hr_ax = mass_et_list = None

if z_values and et_data_dict and star_identifier and 0 < len(star_names) == len(star_sizes) == len(star_colors) == len(star_x) == len(star_y):
    hr_fig, hr_ax, mass_et_list = plot_evol_tracks_isoc_hr(z_values, et_data_dict, isoc_data_dict, star_identifier, star_names, star_sizes, star_colors, star_x, star_y)
    st.pyplot(hr_fig)

    # save full plot
    buf_fullplot = BytesIO()
    hr_fig.savefig(buf_fullplot, format="png", bbox_inches="tight")
    buf_fullplot.seek(0)
    st.download_button("Download Full Plot", data=buf_fullplot, mime="image/png")

# ---------------------------------------------------------------------
# 5) Display range controls (unchanged)
# ---------------------------------------------------------------------
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

    # label evolutionary tracks with mass values
    label_evol_tracks_with_mass(hr_ax, mass_et_list)

    st.pyplot(hr_fig)

    # save the ranged plot
    buf_rangeplot = BytesIO()
    hr_fig.savefig(buf_rangeplot, format="png", bbox_inches="tight")
    buf_rangeplot.seek(0)
    st.download_button("Download Ranged Plot", data=buf_rangeplot, mime="image/png")
