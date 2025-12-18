import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('PDF') # Uncomment if running in an environment without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec  # Import GridSpec
import seaborn as sns
import os
import itertools
import glob
import math

# --- Configuration ---
FILE_PATH = 'parameter_sweep_summary.csv'
# Assume raw data files are in the same directory as the CSV by default
RAW_DATA_DIR = os.path.dirname(FILE_PATH) if os.path.dirname(FILE_PATH) else '.'
OUTPUT_DIR = 'parameter_scan_plots_pdf'
PLOT_FORMAT = 'pdf'  # Choose 'pdf', 'png', 'svg', etc.

# Updated parameter columns for new format
PARAMETER_COLS = [
    'power_dbm',
    'f_mod_hz',
    'f_dev_khz'
]

# Minimum table results mapping
MIN_TABLE_RESULTS = {
    'sensitivity_nT_rtHz': 'Sensitivity',
    'linewidth_hz': 'Linewidth'
}

# Define RESULT_COLS for plots (controls scaling, units, clipping for individual plots)
RESULT_COLS = {
    'sensitivity_nT_rtHz': {'unit': 'nT/√Hz', 'clip': 20, 'log': False, 'format': '%.2f'},
    'linewidth_hz': {'unit': 'MHz', 'scale': 1e-6, 'clip': None, 'log': False, 'format': '%.2f'}
}

# Define ALL_RESULT_COLS_FOR_TABLES for minimum value tables (can have different formatting/scaling)
ALL_RESULT_COLS_FOR_TABLES = {
    'sensitivity_nT_rtHz': {'unit': 'nT/√Hz', 'scale': 1, 'format': '%.3f'},
    'linewidth_hz': {'unit': 'MHz', 'scale': 1e-6, 'format': '%.2f'}
}

# Define rounding rules for parameter columns before grouping/plotting
PARAM_ROUNDING = {
    'power_dbm': 1,  # Round to 1 decimal place
    'f_mod_hz': -2,  # Round to nearest 100 Hz
    'f_dev_khz': 1  # Round to 1 decimal place (already in kHz)
}

# --- Dashboard Specific Configuration ---
# Define the exact 1D plots for the dashboard
# Format: (result_column, parameter_column)
DASHBOARD_1D_PLOTS_SPEC = [
    ('sensitivity_nT_rtHz', 'f_dev_khz'),
    ('sensitivity_nT_rtHz', 'power_dbm'),
    ('sensitivity_nT_rtHz', 'f_mod_hz')
]

# Define the exact 2D plot for the dashboard
# Format: (result_column, (param1_column, param2_column))
DASHBOARD_2D_PLOT_SPEC = ('sensitivity_nT_rtHz', ('power_dbm', 'f_dev_khz'))

# Configuration for the minimum sensitivity ASD plot
MIN_SENS_RESULT_COL = 'sensitivity_nT_rtHz'  # Result column to minimize for ASD plot
ASD_FILENAME_SUFFIX = '_ASD.csv'
ASD_FREQ_COL = 'frequencies'  # Column name for frequency in ASD file
ASD_DATA_COLS_TO_PLOT = {  # Columns to plot from ASD file and their labels/styles
    'asd_hanning': {'label': 'Hann', 'alpha': 0.8, 'linestyle': '--', 'color': 'dimgray', 'zorder': 3},
    # 'asd_boxcar': {'label': 'Boxcar', 'alpha': 0.7, 'linestyle': '-.', 'color': 'blue', 'zorder': 2},
    # 'asd_blackmanharris': {'label': 'Blackman-Harris', 'alpha': 0.7, 'linestyle': ':', 'color': 'orange', 'zorder': 1}
}
ASD_Y_LABEL = "ASD [nT/" + r"$\sqrt{\mathrm{Hz}}$" + "]"
ASD_Y_LIM = [1E-4, 1E3]  # Y-axis limits for ASD plot


# --- Helper Functions ---
def clean_col_name(col_name):
    """Removes units and replaces special chars for filenames/titles. Uses LaTeX for f_mod/f_dev."""
    if col_name == 'f_mod_hz':
        return r'$f_{\mathrm{mod}}$'
    elif col_name == 'f_dev_khz':
        return r'$f_{\mathrm{dev}}$'
    elif col_name == 'power_dbm':
        return 'Power'
    elif col_name == 'sensitivity_nT_rtHz':
        return 'Sensitivity'
    elif col_name == 'linewidth_hz':
        return 'Linewidth'
    elif col_name == 'zc_slope_V_per_Hz':
        return 'ZC Slope'
    name = col_name.replace('_', ' ').title()
    return name


def format_parameter_value_for_table(param_name, value):
    """Formats parameter values for display in tables based on desired precision."""
    if pd.isna(value): return "N/A"
    value = pd.to_numeric(value, errors='coerce')
    if pd.isna(value): return "N/A"

    if param_name == 'power_dbm':
        return f"{value:.1f} dBm"
    elif param_name == 'f_dev_khz':
        # Already in kHz
        if value >= 10:
            return f"{value:.1f} kHz"
        elif value >= 1:
            return f"{value:.2f} kHz"
        else:
            return f"{value * 1000:.0f} Hz"  # Convert to Hz if < 1 kHz
    elif param_name == 'f_mod_hz':
        val_khz = value / 1000.0
        if val_khz >= 1:
            return f"{val_khz:.1f} kHz"
        else:
            return f"{value:.0f} Hz"
    else:
        # Fallback formatting for other parameters
        try:
            if abs(value) > 1000 or abs(value) < 0.01 and value != 0:
                return f"{value:.3e}"
            else:
                return f"{value:.3f}".rstrip('0').rstrip('.')
        except:
            return str(value)


def format_result_value_for_table(result_name, value, info_dict):
    """Formats result values for display in tables based on info_dict."""
    if pd.isna(value): return "N/A"
    value = pd.to_numeric(value, errors='coerce')
    if pd.isna(value): return "N/A"

    info = info_dict.get(result_name, {})
    scale = info.get('scale', 1)
    unit = info.get('unit', '')
    fmt = info.get('format', '%.3g')

    scaled_value = value * scale

    try:
        formatted_value = fmt % scaled_value
    except (TypeError, ValueError):
        formatted_value = f"{scaled_value:.3g}"
    return f"{formatted_value} {unit}"


def round_to_nearest(series, base):
    """Rounds a pandas Series to the nearest multiple of base, handling NaNs."""
    numeric_series = pd.to_numeric(series, errors='coerce')
    rounded_values = np.where(numeric_series.isna(), np.nan, np.round(numeric_series / base) * base)
    return pd.Series(rounded_values, index=series.index)


def get_axis_formatter(param_col, param_unit_suffix):
    """Returns a Matplotlib ticker formatter appropriate for the parameter axis."""
    if param_col == 'power_dbm':
        return mticker.FormatStrFormatter('%.1f')
    elif param_col == 'f_dev_khz':
        # Already in kHz
        return mticker.FuncFormatter(lambda x, pos: f'{int(x)}' if np.isclose(x, round(x)) else f'{x:.1f}')
    elif param_col == 'f_mod_hz' and param_unit_suffix == " [kHz]":
        return mticker.FuncFormatter(lambda x, pos: f'{int(x)}' if np.isclose(x, round(x)) else f'{x:.1f}')
    return mticker.ScalarFormatter()


# --- Plotting Functions ---
def plot_1d_dependence(df, param_col, result_col_info, output_dir, plot_format='pdf'):
    """Plots the mean +/- min/max range of a result vs. one parameter."""
    result_col = list(result_col_info.keys())[0]
    info = list(result_col_info.values())[0]
    if param_col not in df.columns or result_col not in df.columns:
        print(f"Warning: Missing columns for 1D plot: {param_col} or {result_col}")
        return None, None

    # --- Data Prep ---
    df_plot = df[[param_col, result_col]].copy()
    df_plot[param_col] = pd.to_numeric(df_plot[param_col], errors='coerce')
    df_plot[result_col] = pd.to_numeric(df_plot[result_col], errors='coerce')
    df_filtered = df_plot.dropna()
    if df_filtered.empty:
        print(f"Warning: No valid numeric data for {result_col} vs {param_col}. Skipping 1D plot.")
        return None, None

    # --- Aggregation ---
    try:
        grouped_df = df_filtered.groupby(param_col, dropna=True).agg(
            result_mean=(result_col, 'mean'),
            result_min=(result_col, 'min'),
            result_max=(result_col, 'max')
        ).reset_index()
    except Exception as e:
        print(f"Warning: Aggregation failed for {param_col} vs {result_col}: {e}. Skipping 1D plot.")
        return None, None

    if grouped_df.empty:
        print(f"Warning: Grouped data is empty for {result_col} vs {param_col}. Skipping 1D plot.")
        return None, None

    # --- Styling & Scaling ---
    scale_factor = info.get('scale', 1)
    unit = info.get('unit', '')
    clip_val = info.get('clip', None)
    use_log = info.get('log', False)
    result_format = info.get('format', '%.3g')

    # Apply scaling, formatting, clipping
    for col_suffix in ['mean', 'min', 'max']:
        src_col = f'result_{col_suffix}'
        plot_col = f'plot_{col_suffix}'
        grouped_df[plot_col] = pd.to_numeric(grouped_df[src_col], errors='coerce') * scale_factor
        if result_col == 'linewidth_hz' and unit == 'MHz':
            grouped_df[plot_col] = grouped_df[plot_col].round(2)
        if clip_val is not None:
            grouped_df[plot_col] = grouped_df[plot_col].clip(upper=clip_val)

    grouped_df.dropna(subset=[param_col, 'plot_mean', 'plot_min', 'plot_max'], inplace=True)
    if grouped_df.empty:
        print(f"Warning: No numeric data left for plot {result_col} vs {param_col} after processing.")
        return None, None

    # --- X-axis unit conversion ---
    param_unit_suffix = ""
    x_data = grouped_df[param_col].copy()
    param_name_cleaned = clean_col_name(param_col)
    x_data_numeric = pd.to_numeric(x_data, errors='coerce')

    # Handle unit conversions for display
    if not x_data_numeric.isna().all():
        max_val = x_data_numeric.dropna().max()
        if param_col == 'f_mod_hz' and max_val >= 1000:
            x_data_numeric /= 1000
            param_unit_suffix = " [kHz]"
        elif param_col == 'f_dev_khz':
            param_unit_suffix = " [kHz]"  # Already in kHz
        elif param_col == 'power_dbm':
            param_unit_suffix = " [dBm]"
        x_data = x_data_numeric

    x_label = param_name_cleaned + param_unit_suffix
    y_label = f"{clean_col_name(result_col)} [{unit}]"
    title = f"{clean_col_name(result_col)} vs {param_name_cleaned}"

    fig, ax = plt.subplots(figsize=(5, 3.5))
    try:
        # Prepare final data for plotting
        x_plot_final = pd.to_numeric(x_data, errors='coerce')
        y_mean_final = grouped_df['plot_mean']
        y_min_final = grouped_df['plot_min']
        y_max_final = grouped_df['plot_max']

        # Filter out any remaining NaNs and sort
        valid_mask = x_plot_final.notna() & y_mean_final.notna() & y_min_final.notna() & y_max_final.notna()
        if not valid_mask.any():
            raise ValueError("No valid data points left to plot after final filtering")

        sort_indices = x_plot_final[valid_mask].sort_values().index
        x_plot_final_sorted = x_plot_final.loc[sort_indices]
        y_mean_final_sorted = y_mean_final.loc[sort_indices]
        y_min_final_sorted = y_min_final.loc[sort_indices]
        y_max_final_sorted = y_max_final.loc[sort_indices]

        # Plot mean line and shaded min-max region
        ax.plot(x_plot_final_sorted, y_mean_final_sorted, marker='o', linestyle='-', label='Mean', zorder=3)
        ax.fill_between(x_plot_final_sorted, y_min_final_sorted, y_max_final_sorted, alpha=0.3, label='Min-Max Range',
                        color='tab:blue', zorder=2)
        ax.legend()

    except Exception as e:
        print(f"Error plotting 1D data for {title}: {e}. Skipping plot.")
        plt.close(fig)
        return None, None

    # Apply labels, title, and formatting
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Apply formatters
    try:
        if not x_plot_final_sorted.empty:
            x_formatter = get_axis_formatter(param_col, param_unit_suffix)
            if x_formatter: ax.xaxis.set_major_formatter(x_formatter)
    except Exception as e_fmt:
        print(f"Warning: Could not apply x-axis formatter for {param_col}: {e_fmt}")

    try:
        if not y_mean_final_sorted.empty:
            y_formatter = mticker.FormatStrFormatter(result_format)
            ax.yaxis.set_major_formatter(y_formatter)
    except Exception as e_fmt:
        print(f"Warning: Could not apply y-axis formatter for {result_col}: {e_fmt}")

    if use_log:
        if (y_mean_final_sorted[y_mean_final_sorted > 0]).empty or (y_min_final_sorted[y_min_final_sorted > 0]).empty:
            print(f"Warning: Cannot set log scale for {result_col} vs {param_col} due to non-positive values.")
        else:
            ax.set_yscale('log')

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save figure
    safe_res_name = result_col.replace('/', '_').replace('\\', '_').replace(' ', '_')
    safe_param_name = param_col.replace('/', '_').replace('\\', '_').replace(' ', '_')
    filename = f"{safe_res_name}_vs_{safe_param_name}.{plot_format}"
    filepath = os.path.join(output_dir, filename)
    try:
        plt.savefig(filepath, format=plot_format, bbox_inches='tight')
        print(f"Saved: {filepath}")
    except Exception as e:
        print(f"Error saving figure {filepath}: {e}")

    plt.close(fig)
    return None, None


def plot_2d_dependence(df, param1_col, param2_col, result_col_info, output_dir, plot_format='pdf'):
    """Plots a 2D heatmap of the mean result vs. two parameters."""
    result_col = list(result_col_info.keys())[0]
    info = list(result_col_info.values())[0]
    if not all(p in df.columns for p in [param1_col, param2_col, result_col]):
        print(f"Warning: Missing columns for 2D plot involving {param1_col}, {param2_col}, {result_col}")
        return None, None

    # --- Data Prep ---
    df_plot = df[[param1_col, param2_col, result_col]].copy()
    df_plot[param1_col] = pd.to_numeric(df_plot[param1_col], errors='coerce')
    df_plot[param2_col] = pd.to_numeric(df_plot[param2_col], errors='coerce')
    df_plot[result_col] = pd.to_numeric(df_plot[result_col], errors='coerce')
    df_filtered = df_plot.dropna()
    if df_filtered.empty:
        print(f"Warning: No valid numeric data for {result_col} vs {param1_col} and {param2_col}. Skipping 2D plot.")
        return None, None

    # --- Aggregation ---
    try:
        grouped_df = df_filtered.groupby([param1_col, param2_col], dropna=True).agg(
            mean_result=(result_col, 'mean')
        ).reset_index()
    except Exception as e:
        print(f"Warning: Grouping failed for {param1_col}, {param2_col} vs {result_col}: {e}. Skipping 2D plot.")
        return None, None

    if grouped_df.empty:
        print(f"Warning: Grouped data is empty for {result_col} vs {param1_col}, {param2_col}. Skipping 2D plot.")
        return None, None

    # --- Styling & Scaling ---
    scale_factor = info.get('scale', 1)
    unit = info.get('unit', '')
    clip_val = info.get('clip', None)
    result_format = info.get('format', '%.3g')

    # Apply scaling and clipping
    grouped_df['plot_result'] = pd.to_numeric(grouped_df['mean_result'], errors='coerce') * scale_factor
    if result_col == 'linewidth_hz' and unit == 'MHz':
        grouped_df['plot_result'] = grouped_df['plot_result'].round(2)
    if clip_val is not None:
        grouped_df['plot_result'] = grouped_df['plot_result'].clip(upper=clip_val)

    # --- Parameter unit conversion for PIVOT axes ---
    def convert_axis_for_pivot(df_group, param_col_name):
        param_unit_suffix = ""
        param_name_cleaned = clean_col_name(param_col_name)
        data_for_pivot = pd.to_numeric(df_group[param_col_name], errors='coerce').copy()
        if not data_for_pivot.isna().all():
            max_val = data_for_pivot.dropna().max()
            if param_col_name == 'f_mod_hz' and max_val >= 1000:
                data_for_pivot /= 1000
                param_unit_suffix = " [kHz]"
            elif param_col_name == 'f_dev_khz':
                param_unit_suffix = " [kHz]"  # Already in kHz
            elif param_col_name == 'power_dbm':
                param_unit_suffix = " [dBm]"
        return data_for_pivot, param_name_cleaned, param_unit_suffix

    grouped_df['pivot_param1'], param1_name_cleaned, param1_unit_suffix = convert_axis_for_pivot(grouped_df, param1_col)
    grouped_df['pivot_param2'], param2_name_cleaned, param2_unit_suffix = convert_axis_for_pivot(grouped_df, param2_col)

    param1_label = param1_name_cleaned + param1_unit_suffix
    param2_label = param2_name_cleaned + param2_unit_suffix

    # Drop rows where pivot index/columns or values would be NaN
    grouped_df.dropna(subset=['pivot_param1', 'pivot_param2', 'plot_result'], inplace=True)
    if grouped_df.empty:
        print(f"Warning: No valid data remains for pivot table ({param1_label} vs {param2_label}). Skipping 2D plot.")
        return None, None

    # --- Pivot Data ---
    try:
        pivot_table = pd.pivot_table(grouped_df, index='pivot_param2', columns='pivot_param1', values='plot_result',
                                     aggfunc='mean')
    except Exception as e:
        print(f"Error during pivot for {param1_label} vs {param2_label}: {e}. Skipping 2D plot.")
        return None, None

    if pivot_table.empty:
        print(f"Warning: Pivot table is empty for {param1_label} vs {param2_label}. Skipping 2D plot.")
        return None, None

    # --- Plot Heatmap ---
    cbar_label = f"Mean {clean_col_name(result_col)} [{unit}]"
    title = f"{clean_col_name(result_col)} vs {param1_name_cleaned} and {param2_name_cleaned}"

    fig, ax = plt.subplots(figsize=(6, 4.5))
    try:
        cbar_formatter = mticker.FormatStrFormatter(result_format)
        # Ensure vmin/vmax are sensible
        valid_data = pivot_table.unstack().dropna()
        vmin = valid_data.min() if not valid_data.empty else None
        vmax = valid_data.max() if not valid_data.empty else None
        if vmin == vmax and vmin is not None:
            vmin -= 0.1 * abs(vmin) if vmin != 0 else 0.1
            vmax += 0.1 * abs(vmax) if vmax != 0 else 0.1

        sns.heatmap(pivot_table, cmap="viridis", annot=False, fmt=".2f",
                    ax=ax,
                    cbar_kws={'label': cbar_label, 'format': cbar_formatter},
                    vmin=vmin, vmax=vmax)

        ax.set_xlabel(param1_label)
        ax.set_ylabel(param2_label)
        ax.set_title(title)

        # Set ticks and labels
        try:
            # X-axis
            x_tick_values = pivot_table.columns.values
            x_fmt = get_axis_formatter(param1_col, param1_unit_suffix)
            x_tick_labels = [x_fmt(val, None) for val in x_tick_values]
            ax.set_xticks(np.arange(len(x_tick_values)) + 0.5)
            ax.set_xticklabels(x_tick_labels)
        except Exception as e_fmt_x:
            print(f"Warning: Could not set x-axis ticks/labels for 2D plot {param1_col}: {e_fmt_x}")

        try:
            # Y-axis
            y_tick_values = pivot_table.index.values
            y_fmt = get_axis_formatter(param2_col, param2_unit_suffix)
            y_tick_labels = [y_fmt(val, None) for val in y_tick_values]
            ax.set_yticks(np.arange(len(y_tick_values)) + 0.5)
            ax.set_yticklabels(y_tick_labels)
        except Exception as e_fmt_y:
            print(f"Warning: Could not set y-axis ticks/labels for 2D plot {param2_col}: {e_fmt_y}")

        # Rotate tick labels
        plt.setp(ax.get_xticklabels(), rotation=90)
        plt.setp(ax.get_yticklabels(), rotation=0)

    except Exception as e:
        print(f"Error plotting heatmap for {title}: {e}. Skipping plot.")
        plt.close(fig)
        return None, None

    plt.tight_layout()

    # Save figure
    safe_res_name = result_col.replace('/', '_').replace('\\', '_').replace(' ', '_')
    safe_param1_name = param1_col.replace('/', '_').replace('\\', '_').replace(' ', '_')
    safe_param2_name = param2_col.replace('/', '_').replace('\\', '_').replace(' ', '_')
    filename = f"{safe_res_name}_vs_{safe_param1_name}_and_{safe_param2_name}.{plot_format}"
    filepath = os.path.join(output_dir, filename)
    try:
        plt.savefig(filepath, format=plot_format, bbox_inches='tight')
        print(f"Saved: {filepath}")
    except Exception as e:
        print(f"Error saving figure {filepath}: {e}")

    plt.close(fig)
    return None, None


def create_min_value_table(df, result_col, display_name, param_cols, result_format_info, ax, title_prefix="Minimum"):
    """Creates a table on the given axes showing the minimum value and parameters."""
    if result_col not in df.columns:
        ax.text(0.5, 0.5, f"Result column\n'{result_col}'\nnot found.", ha='center', va='center', fontsize=9,
                color='red')
        ax.set_title(f"{title_prefix} {display_name} (Error)", fontsize=10)
        ax.axis('off');
        return None

    df_numeric = df.copy()
    df_numeric[result_col] = pd.to_numeric(df_numeric[result_col], errors='coerce')
    df_numeric = df_numeric.dropna(subset=[result_col])

    if df_numeric.empty:
        ax.text(0.5, 0.5, f"No valid numeric data\nfor '{result_col}'.", ha='center', va='center', fontsize=9,
                color='red')
        ax.set_title(f"{title_prefix} {display_name} (No Data)", fontsize=10)
        ax.axis('off');
        return None

    try:
        min_idx = df_numeric[result_col].idxmin()
        min_row = df.loc[min_idx]
        min_value = min_row[result_col]

        table_data = []
        formatted_min_value = format_result_value_for_table(result_col, min_value, result_format_info)
        table_data.append([f"Min {display_name}", formatted_min_value])

        # Add rows for parameters
        for p_col in param_cols:
            if p_col in min_row.index:
                p_name_clean = clean_col_name(p_col)
                p_value = min_row[p_col]
                formatted_p_value = format_parameter_value_for_table(p_col, p_value)
                table_data.append([p_name_clean, formatted_p_value])
            else:
                table_data.append([clean_col_name(p_col), "N/A"])

        # Create and style the table
        the_table = ax.table(cellText=table_data, loc='center', cellLoc='left', colWidths=[0.55, 0.45])
        the_table.auto_set_font_size(False);
        the_table.set_fontsize(9);
        the_table.scale(1, 1.2)

        # Style
        for key, cell in the_table.get_celld().items():
            cell.set_edgecolor('none')
            if key[1] == 0:
                cell.set_text_props(weight='bold')
            else:
                cell.set_text_props(ha='left')

        ax.set_title(f"{title_prefix} {display_name}", fontsize=10, weight='bold', pad=2)
        ax.axis('off')
        return min_idx

    except Exception as e:
        print(f"Error creating table for {result_col}: {e}")
        import traceback
        traceback.print_exc()
        ax.text(0.5, 0.5, f"Error creating table for\n'{result_col}'.", ha='center', va='center', fontsize=9,
                color='red')
        ax.set_title(f"{title_prefix} {display_name} (Error)", fontsize=10)
        ax.axis('off')
        return None


# Helper functions for dashboard (simplified versions)
def _add_1d_plot_to_dashboard(ax, df, result_col, param_col, result_cols_info):
    """Internal helper to add a single 1D plot to a dashboard axis."""
    info = result_cols_info.get(result_col, {})
    if param_col not in df.columns or result_col not in df.columns:
        ax.text(0.5, 0.5, f"Data missing for\n{clean_col_name(result_col)}\nvs {clean_col_name(param_col)}",
                ha='center', va='center', fontsize=8, color='grey')
        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.set_title(f"{clean_col_name(result_col)} vs {clean_col_name(param_col)} (Missing Data)", fontsize=9);
        return False

    scale_factor = info.get('scale', 1)
    unit = info.get('unit', '')
    clip_val = info.get('clip', None)
    use_log = info.get('log', False)
    result_format = info.get('format', '%.3g')

    df_plot = df[[param_col, result_col]].copy()
    df_plot[param_col] = pd.to_numeric(df_plot[param_col], errors='coerce')
    df_plot[result_col] = pd.to_numeric(df_plot[result_col], errors='coerce')
    df_filtered = df_plot.dropna()

    if df_filtered.empty:
        ax.text(0.5, 0.5, f"No valid data for\n{clean_col_name(result_col)}\nvs {clean_col_name(param_col)}",
                ha='center', va='center', fontsize=8, color='grey')
        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.set_title(f"{clean_col_name(result_col)} vs {clean_col_name(param_col)} (No Data)", fontsize=9);
        return False

    try:
        # Aggregate
        grouped_df = df_filtered.groupby(param_col, dropna=True).agg(
            result_mean=(result_col, 'mean'),
            result_min=(result_col, 'min'),
            result_max=(result_col, 'max')
        ).reset_index()

        if grouped_df.empty: raise ValueError("Grouping resulted in empty dataframe")

        # Apply scaling, clipping
        for suffix in ['mean', 'min', 'max']:
            src, target = f'result_{suffix}', f'plot_{suffix}'
            grouped_df[target] = pd.to_numeric(grouped_df[src], errors='coerce') * scale_factor
            if result_col == 'linewidth_hz' and unit == 'MHz': grouped_df[target] = grouped_df[target].round(2)
            if clip_val is not None: grouped_df[target] = grouped_df[target].clip(upper=clip_val)

        grouped_df.dropna(subset=[param_col, 'plot_mean', 'plot_min', 'plot_max'], inplace=True)
        if grouped_df.empty: raise ValueError("No data after NaN drop post-scaling")

        # X-axis conversion
        param_unit_suffix = ""
        x_data = grouped_df[param_col].copy()
        param_name_cleaned = clean_col_name(param_col)
        x_data_numeric = pd.to_numeric(x_data, errors='coerce')
        if not x_data_numeric.isna().all():
            max_val = x_data_numeric.dropna().max()
            if param_col == 'f_mod_hz' and max_val >= 1000:
                x_data_numeric /= 1000
                param_unit_suffix = " [kHz]"
            elif param_col == 'f_dev_khz':
                param_unit_suffix = " [kHz]"
            elif param_col == 'power_dbm':
                param_unit_suffix = " [dBm]"
            x_data = x_data_numeric

        # Plotting
        x_plot = pd.to_numeric(x_data, errors='coerce')
        y_mean, y_min, y_max = grouped_df['plot_mean'], grouped_df['plot_min'], grouped_df['plot_max']
        mask = x_plot.notna() & y_mean.notna() & y_min.notna() & y_max.notna()
        if not mask.any(): raise ValueError("No valid points after final filter")

        sort_idx = x_plot[mask].sort_values().index
        x_sorted, y_mean_sorted = x_plot.loc[sort_idx], y_mean.loc[sort_idx]
        y_min_sorted, y_max_sorted = y_min.loc[sort_idx], y_max.loc[sort_idx]

        ax.plot(x_sorted, y_mean_sorted, marker='.', markersize=5, linestyle='-', label='Mean', zorder=3)
        ax.fill_between(x_sorted, y_min_sorted, y_max_sorted, alpha=0.3, label='Min-Max Range', color='tab:blue',
                        zorder=2)
        ax.legend(fontsize=8)

        # Formatting
        ax.set_xlabel(param_name_cleaned + param_unit_suffix, fontsize=9)
        ax.set_ylabel(f"{clean_col_name(result_col)} [{unit}]", fontsize=9)
        ax.set_title(f"{clean_col_name(result_col)} vs {param_name_cleaned}", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)

        if not x_sorted.empty:
            x_fmt = get_axis_formatter(param_col, param_unit_suffix)
            if x_fmt: ax.xaxis.set_major_formatter(x_fmt)
            if len(x_sorted) > 10:
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, prune='both'))

        if not y_mean_sorted.empty:
            y_fmt = mticker.FormatStrFormatter(result_format)
            if y_fmt: ax.yaxis.set_major_formatter(y_fmt)
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, prune='both'))

        if use_log:
            if (y_mean_sorted > 0).all() and (y_min_sorted > 0).all():
                ax.set_yscale('log')
            else:
                print(f"    Warning: Cannot set log scale for dashboard {result_col} vs {param_col}")

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        return True

    except Exception as e:
        print(
            f"    Error generating dashboard 1D plot for {result_col} vs {param_col}: {type(e).__name__} - {e}. Skipping subplot.")
        ax.clear();
        ax.text(0.5, 0.5,
                f"Plot failed\n({clean_col_name(result_col)} vs {clean_col_name(param_col)})\n{type(e).__name__}",
                ha='center', va='center', fontsize=8, color='red');
        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.set_title(f"{clean_col_name(result_col)} vs {clean_col_name(param_col)} (Plot Failed)", fontsize=9);
        return False


def _add_2d_plot_to_dashboard(ax, df, result_col, param1_col, param2_col, result_cols_info):
    """Internal helper to add a single 2D plot to a dashboard axis."""
    info = result_cols_info.get(result_col, {})
    if not all(c in df.columns for c in [param1_col, param2_col, result_col]):
        ax.text(0.5, 0.5,
                f"Data missing for\n{clean_col_name(result_col)}\nvs {clean_col_name(param1_col)}\n& {clean_col_name(param2_col)}",
                ha='center', va='center', fontsize=8, color='grey')
        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.set_title(f"2D Plot (Missing Data)", fontsize=9);
        return False

    scale_factor, unit = info.get('scale', 1), info.get('unit', '')
    clip_val, result_format = info.get('clip', None), info.get('format', '%.3g')

    df_plot = df[[param1_col, param2_col, result_col]].copy()
    df_plot[param1_col] = pd.to_numeric(df_plot[param1_col], errors='coerce')
    df_plot[param2_col] = pd.to_numeric(df_plot[param2_col], errors='coerce')
    df_plot[result_col] = pd.to_numeric(df_plot[result_col], errors='coerce')
    df_filtered = df_plot.dropna()

    if df_filtered.empty:
        ax.text(0.5, 0.5,
                f"No valid data for\n{clean_col_name(result_col)}\nvs {clean_col_name(param1_col)}\n& {clean_col_name(param2_col)}",
                ha='center', va='center', fontsize=8, color='grey')
        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.set_title(f"2D Plot (No Data)", fontsize=9);
        return False

    try:
        # Aggregation
        grouped_df = df_filtered.groupby([param1_col, param2_col], dropna=True).agg(
            mean_result=(result_col, 'mean')
        ).reset_index()

        grouped_df['plot_result'] = pd.to_numeric(grouped_df['mean_result'], errors='coerce') * scale_factor
        if result_col == 'linewidth_hz' and unit == 'MHz': grouped_df['plot_result'] = grouped_df['plot_result'].round(
            2)
        if clip_val is not None: grouped_df['plot_result'] = grouped_df['plot_result'].clip(upper=clip_val)

        # Pivot Prep
        def convert_axis_for_pivot(df_group, param_col_name):
            param_unit_suffix = ""
            param_name_cleaned = clean_col_name(param_col_name)
            data_for_pivot = pd.to_numeric(df_group[param_col_name], errors='coerce').copy()
            if not data_for_pivot.isna().all():
                max_val = data_for_pivot.dropna().max()
                if param_col_name == 'f_mod_hz' and max_val >= 1000:
                    data_for_pivot /= 1000
                    param_unit_suffix = " [kHz]"
                elif param_col_name == 'f_dev_khz':
                    param_unit_suffix = " [kHz]"
                elif param_col_name == 'power_dbm':
                    param_unit_suffix = " [dBm]"
            return data_for_pivot, param_name_cleaned, param_unit_suffix

        grouped_df['pivot_p1'], p1_clean, p1_suffix = convert_axis_for_pivot(grouped_df, param1_col)
        grouped_df['pivot_p2'], p2_clean, p2_suffix = convert_axis_for_pivot(grouped_df, param2_col)

        grouped_df.dropna(subset=['pivot_p1', 'pivot_p2', 'plot_result'], inplace=True)
        if grouped_df.empty: raise ValueError("No data for pivot")

        pivot_table = pd.pivot_table(grouped_df, index='pivot_p2', columns='pivot_p1', values='plot_result',
                                     aggfunc='mean')
        if pivot_table.empty: raise ValueError("Pivot table empty")

        # Plotting
        cbar_label, cbar_fmt = f"{clean_col_name(result_col)} [{unit}]", mticker.FormatStrFormatter(result_format)
        valid_data = pivot_table.unstack().dropna()
        vmin = valid_data.min() if not valid_data.empty else None
        vmax = valid_data.max() if not valid_data.empty else None
        if vmin == vmax and vmin is not None:
            vmin -= 0.1 * abs(vmin) if vmin != 0 else 0.1
            vmax += 0.1 * abs(vmax) if vmax != 0 else 0.1

        sns.heatmap(pivot_table, cmap="viridis", annot=False, fmt=".2f", ax=ax,
                    cbar_kws={'label': cbar_label, 'format': cbar_fmt},
                    vmin=vmin, vmax=vmax)

        # Set colorbar label size
        ax.figure.axes[-1].yaxis.label.set_size(9)
        ax.figure.axes[-1].tick_params(labelsize=8)

        ax.set_xlabel(p1_clean + p1_suffix, fontsize=9);
        ax.set_ylabel(p2_clean + p2_suffix, fontsize=9);
        ax.set_title(f"{clean_col_name(result_col)} vs {p1_clean} & {p2_clean}", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)

        try:
            # X-axis
            x_tick_values = pivot_table.columns.values
            x_fmt = get_axis_formatter(param1_col, p1_suffix)
            x_tick_labels = [x_fmt(val, None) for val in x_tick_values]
            ax.set_xticks(np.arange(len(x_tick_values)) + 0.5)
            ax.set_xticklabels(x_tick_labels, fontsize=8)
        except Exception as e_fmt_x:
            print(f"    Warning: Could not set x-axis ticks/labels for dashboard 2D plot {param1_col}: {e_fmt_x}")

        try:
            # Y-axis
            y_tick_values = pivot_table.index.values
            y_fmt = get_axis_formatter(param2_col, p2_suffix)
            y_tick_labels = [y_fmt(val, None) for val in y_tick_values]
            ax.set_yticks(np.arange(len(y_tick_values)) + 0.5)
            ax.set_yticklabels(y_tick_labels, fontsize=8)
        except Exception as e_fmt_y:
            print(f"    Warning: Could not set y-axis ticks/labels for dashboard 2D plot {param2_col}: {e_fmt_y}")

        plt.setp(ax.get_xticklabels(), rotation=90)
        plt.setp(ax.get_yticklabels(), rotation=0)
        return True

    except Exception as e:
        print(
            f"    Error generating dashboard 2D plot for {result_col} vs {param1_col}, {param2_col}: {type(e).__name__} - {e}. Skipping subplot.")
        ax.clear();
        ax.text(0.5, 0.5,
                f"Plot failed\n({clean_col_name(result_col)} vs {clean_col_name(param1_col)}, {clean_col_name(param2_col)})\n{type(e).__name__}",
                ha='center', va='center', fontsize=8, color='red');
        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.set_title(f"2D Plot (Plot Failed)", fontsize=9);
        return False


def _find_min_result_row(df, result_col):
    """Finds the index of the row with the minimum value in the specified result column."""
    if result_col not in df.columns:
        print(f"Error: Minimum result column '{result_col}' not found.")
        return None

    df_numeric = df.copy()
    df_numeric[result_col] = pd.to_numeric(df_numeric[result_col], errors='coerce')
    df_numeric = df_numeric.dropna(subset=[result_col])

    if df_numeric.empty:
        print(f"Warning: No numeric data found for minimum value search in '{result_col}'.")
        return None

    try:
        min_idx = df_numeric[result_col].idxmin()
        return min_idx
    except Exception as e:
        print(f"Error finding minimum value index for '{result_col}': {e}")
        return None


def _construct_asd_filename(min_row, parameter_cols):
    """Constructs the expected ASD filename from the minimum sensitivity row data."""
    # NOTE: This function needs to be adapted to the new data format and file naming convention
    # For now, returning None as we don't know the new naming convention
    print("Warning: ASD filename construction not implemented for new data format.")
    return None


def calculate_asd_noise_floor(freqs, asd_values, f1, f2, filter_frequencies=None, filter_intervals=None):
    """Calculate the noise floor of the ASD within a specified frequency range."""
    # Convert inputs to numpy arrays
    freqs = np.asarray(freqs)
    asd_values = np.asarray(asd_values)

    # Input Validation
    if freqs.shape != asd_values.shape or freqs.ndim != 1:
        print("Warning: Frequencies and ASD values must be 1D arrays of the same shape.")
        return float('nan'), 0.0
    if f1 >= f2:
        print(f"Warning: f1 ({f1}) must be less than f2 ({f2}).")
        return float('nan'), 0.0

    # Initial filtering for the [f1, f2] range
    initial_range_mask = (freqs >= f1) & (freqs <= f2)
    freqs_in_range = freqs[initial_range_mask]
    asd_values_in_range = asd_values[initial_range_mask]

    if len(freqs_in_range) == 0:
        print(f"Warning: No data points found in the initial range [{f1}, {f2}] Hz.")
        return float('nan'), 0.0

    current_freqs = freqs_in_range.copy()
    current_asd = asd_values_in_range.copy()
    remaining_indices = np.ones(len(current_freqs), dtype=bool)

    # Filter specific frequencies
    if filter_frequencies is not None and len(filter_frequencies) > 0:
        filter_freqs_array = np.asarray(filter_frequencies)
        df = np.median(np.diff(freqs)) if len(freqs) > 1 else 1.0
        tolerance = df / 2.1
        indices_to_remove = np.zeros(len(current_freqs), dtype=bool)
        for filt_f in filter_freqs_array:
            indices_to_remove |= (np.abs(current_freqs - filt_f) < tolerance)
        remaining_indices &= ~indices_to_remove
        print(f"  Filtering frequencies around: {filter_frequencies}. Points removed: {np.sum(indices_to_remove)}")

    # Filter frequency intervals
    if filter_intervals is not None:
        for (start_freq, end_freq) in filter_intervals:
            if start_freq >= end_freq:
                print(f"  Warning: Skipping invalid filter interval [{start_freq}, {end_freq}]")
                continue
            interval_mask = (current_freqs < start_freq) | (current_freqs > end_freq)
            points_removed_in_interval = np.sum(remaining_indices & ~interval_mask)
            remaining_indices &= interval_mask
            print(f"  Filtering interval: [{start_freq}, {end_freq}]. Points removed: {points_removed_in_interval}")

    # Apply the combined filter
    final_freqs = current_freqs[remaining_indices]
    final_asd_values = current_asd[remaining_indices]
    asd_uncertainty = np.std(final_asd_values)

    # Compute mean ASD
    if len(final_freqs) == 0:
        print(f"Warning: No data points remaining in range [{f1}, {f2}] Hz after filtering.")
        mean_asd = float('nan')
        equivalent_bandwidth = 0.0
    else:
        mean_asd = np.mean(final_asd_values)
        equivalent_bandwidth = final_freqs.max() - final_freqs.min() if len(final_freqs) > 1 else 0.0

    return mean_asd, asd_uncertainty, equivalent_bandwidth


def _add_asd_plot_to_dashboard(ax, asd_filepath):
    """Adds the ASD plot from a given file path to the dashboard axis."""
    noise_floor = np.nan
    noise_floor_freq_range = (100, 200)  # Hz
    noise_floor_filter_freqs = [100, 101, 149, 150, 151, 199, 200, 201]  # Harmonics of 50 Hz

    plot_title = "Min. Sensitivity ASD"

    if asd_filepath and os.path.exists(asd_filepath):
        print(f"    Attempting to plot ASD file: {os.path.basename(asd_filepath)}")
        try:
            # Try reading with common separators
            try:
                df_asd = pd.read_csv(asd_filepath, sep='\t', comment='#', skipinitialspace=True)
                if ASD_FREQ_COL not in df_asd.columns and df_asd.index.name == ASD_FREQ_COL:
                    df_asd.reset_index(inplace=True)
                elif ASD_FREQ_COL not in df_asd.columns and ASD_FREQ_COL not in df_asd.index.name:
                    df_asd = pd.read_csv(asd_filepath, sep='\t', comment='#', skipinitialspace=True, index_col=None)
            except (ValueError, pd.errors.ParserError):
                try:
                    df_asd = pd.read_csv(asd_filepath, sep=',', comment='#', skipinitialspace=True)
                    if ASD_FREQ_COL not in df_asd.columns and df_asd.index.name == ASD_FREQ_COL:
                        df_asd.reset_index(inplace=True)
                    elif ASD_FREQ_COL not in df_asd.columns and ASD_FREQ_COL not in df_asd.index.name:
                        df_asd = pd.read_csv(asd_filepath, sep=',', comment='#', skipinitialspace=True, index_col=None)
                except Exception as e_read:
                    raise ValueError(f"Could not read ASD CSV: {e_read}")

            if ASD_FREQ_COL not in df_asd.columns:
                raise ValueError(
                    f"Frequency column '{ASD_FREQ_COL}' not found in ASD file columns: {df_asd.columns.tolist()}")

            # Convert columns to numeric
            freq_data = pd.to_numeric(df_asd[ASD_FREQ_COL], errors='coerce')
            asd_plot_data = {}
            plot_possible = False
            first_asd_col_key = None

            for col_key, plot_opts in ASD_DATA_COLS_TO_PLOT.items():
                if col_key in df_asd.columns:
                    data = pd.to_numeric(df_asd[col_key], errors='coerce')
                    valid_mask = freq_data.notna() & data.notna() & (data > 0)
                    if valid_mask.any():
                        asd_plot_data[col_key] = {'freq': freq_data[valid_mask], 'asd': data[valid_mask],
                                                  'opts': plot_opts}
                        plot_possible = True
                        if first_asd_col_key is None:
                            first_asd_col_key = col_key
                else:
                    print(f"      Info: ASD data column '{col_key}' not found in file.")

            if not plot_possible:
                raise ValueError("No valid data found in configured ASD columns.")

            # Calculate Noise Floor
            if first_asd_col_key and first_asd_col_key in asd_plot_data:
                print(
                    f"    Calculating noise floor from '{first_asd_col_key}' data between {noise_floor_freq_range[0]}-{noise_floor_freq_range[1]} Hz")
                calc_freqs = asd_plot_data[first_asd_col_key]['freq'].to_numpy()
                calc_asd_vals = asd_plot_data[first_asd_col_key]['asd'].to_numpy()
                noise_floor, noise_uncertainty, _ = calculate_asd_noise_floor(
                    calc_freqs,
                    calc_asd_vals,
                    f1=noise_floor_freq_range[0],
                    f2=noise_floor_freq_range[1],
                    filter_frequencies=noise_floor_filter_freqs
                )
                if pd.notna(noise_floor):
                    print(f"      Calculated noise floor: {noise_floor:.3f} nT/sqrt(Hz)")
                    # Update plot title
                    if noise_floor < 1.0:
                        plot_title = f"Min. Sensitivity ASD\nNoise Floor ({noise_floor_freq_range[0]}-{noise_floor_freq_range[1]} Hz): {noise_floor * 1000:.1f} ± {noise_uncertainty * 1000:.1f} pT/" + "$\sqrt{Hz}$"
                    else:
                        plot_title = f"Min. Sensitivity ASD\nNoise Floor ({noise_floor_freq_range[0]}-{noise_floor_freq_range[1]} Hz): {noise_floor:.2f} ± {noise_uncertainty:.2f} nT/" + "$\sqrt{Hz}$"
                else:
                    print("      Noise floor calculation failed (NaN result).")
                    plot_title = "Min. Sensitivity ASD\n(Noise Floor Calc Failed)"
            else:
                print("    Skipping noise floor calculation: No suitable ASD data column found.")
                plot_title = "Min. Sensitivity ASD\n(Noise Floor Calc Skipped)"

            # Plot the ASD data
            for col, data_dict in asd_plot_data.items():
                ax.plot(data_dict['freq'], data_dict['asd'], **data_dict['opts'])

            # Plot noise floor line
            if pd.notna(noise_floor):
                ax.axhline(noise_floor, color='red', linestyle='--', linewidth=1,
                           label=f'Noise Floor ({noise_floor * 1000:.1f} ± {noise_uncertainty * 1000:.1f} pT/√Hz)' if noise_floor < 1 else f'Noise Floor ({noise_floor:.2f} nT/√Hz)',
                           zorder=10)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel("Frequency [Hz]", fontsize=9)
            ax.set_ylabel(ASD_Y_LABEL, fontsize=9)
            ax.set_title(plot_title, fontsize=9, pad=15)

            if ASD_Y_LIM:
                ax.set_ylim(ASD_Y_LIM)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.tick_params(axis='both', which='major', labelsize=8)

            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=7, loc='lower left')

            return True

        except Exception as e:
            print(f"    Error processing/plotting ASD file {os.path.basename(asd_filepath)}: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            ax.clear()
            ax.text(0.5, 0.5, f"ASD Plot Failed\nFile: {os.path.basename(asd_filepath)}\nError: {type(e).__name__}",
                    ha='center', va='center', fontsize=8, color='red')
            ax.set_xticks([]);
            ax.set_yticks([]);
            ax.set_title("Min. Sensitivity ASD (Error)", fontsize=9)
            return False
    else:
        print("    ASD file not found or path not provided.")
        ax.text(0.5, 0.5, "ASD file for\nminimum sensitivity\nnot found.", ha='center', va='center', fontsize=8,
                color='grey')
        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.set_title("Min. Sensitivity ASD (Not Found)", fontsize=9)
        return False


def _add_combined_tables_to_dashboard(ax_main, df, min_table_results, all_result_formats, parameter_cols):
    """Adds multiple minimum value tables vertically stacked onto a single main axis."""
    ax_main.axis('off')
    num_tables = len(min_table_results)
    if num_tables == 0:
        ax_main.text(0.5, 0.5, "(No Tables Defined)", ha='center', va='center', fontsize=10, color='grey')
        return {}

    # Create inner GridSpec
    gs_inner = gridspec.GridSpecFromSubplotSpec(num_tables, 1, subplot_spec=ax_main, hspace=0.3)

    min_indices = {}

    table_count = 0
    for i, (res_col, display_name) in enumerate(min_table_results.items()):
        ax_table = plt.subplot(gs_inner[i])
        print(f"    Creating table for Minimum {display_name} in combined area")
        min_idx = create_min_value_table(df, res_col, display_name, parameter_cols, all_result_formats, ax_table)
        if min_idx is not None:
            min_indices[res_col] = min_idx
            table_count += 1

    if table_count == 0:
        for i in range(num_tables): plt.subplot(gs_inner[i]).clear(); plt.subplot(gs_inner[i]).axis('off');
        ax_main.text(0.5, 0.5, "(Table Creation Failed)", ha='center', va='center', fontsize=10, color='red')

    return min_indices


def create_dashboard(df, df_original,
                     varying_params, result_cols_info,
                     dashboard_1d_plots_spec, dashboard_2d_plot_spec,
                     min_table_results, all_result_formats, param_cols_config,
                     output_dir, plot_format='pdf'):
    """Creates a dashboard figure with specifically configured plots and tables."""
    print("\n--- Creating Dashboard ---")

    # Determine what can be plotted/tabulated
    valid_1d_plots = []
    for res_col, param_col in dashboard_1d_plots_spec:
        if res_col in result_cols_info and param_col in varying_params and res_col in df.columns and param_col in df.columns:
            valid_1d_plots.append((res_col, param_col))
        else:
            print(
                f"  Skipping dashboard 1D plot: '{res_col}' vs '{param_col}' (missing column, config, or param not varying).")

    valid_2d_plot = None
    if dashboard_2d_plot_spec:
        res_col_2d, (p1, p2) = dashboard_2d_plot_spec
        if res_col_2d in result_cols_info and p1 in varying_params and p2 in varying_params and res_col_2d in df.columns and p1 in df.columns and p2 in df.columns:
            valid_2d_plot = dashboard_2d_plot_spec
        else:
            print(
                f"  Skipping dashboard 2D plot: '{res_col_2d}' vs '{p1}' & '{p2}' (missing column, config, or param not varying).")

    valid_min_tables = {res: name for res, name in min_table_results.items() if res in df.columns}

    # Find Minimum Sensitivity Row and ASD File Path
    min_sens_idx = None
    min_sens_row = None
    asd_filepath = None
    min_sens_result_col_actual = MIN_SENS_RESULT_COL
    if min_sens_result_col_actual and min_sens_result_col_actual in df.columns:
        min_sens_idx = _find_min_result_row(df, min_sens_result_col_actual)
        if min_sens_idx is not None and min_sens_idx in df_original.index:
            min_sens_row = df_original.loc[min_sens_idx]
            asd_filename = _construct_asd_filename(min_sens_row, PARAMETER_COLS)
            if asd_filename:
                potential_path = os.path.join(RAW_DATA_DIR, asd_filename)
                if os.path.exists(potential_path):
                    asd_filepath = potential_path
                    print(f"  Found ASD file for min sensitivity: {asd_filename}")
                else:
                    print(f"  Constructed ASD filename '{asd_filename}' but file not found at '{potential_path}'")
            else:
                print("  Could not construct ASD filename from minimum sensitivity row.")
        elif min_sens_idx is not None:
            print(
                f"  Warning: Minimum sensitivity index {min_sens_idx} found but not present in original DataFrame index.")
        else:
            print(f"  Could not find minimum sensitivity row for column '{min_sens_result_col_actual}'.")

    # Set up figure layout
    nrows, ncols = 2, 3
    fig_width = ncols * 5
    fig_height = nrows * 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.flatten()

    plot_idx = 0

    # Add 1D plots to the first row
    print("  Adding 1D plots...")
    num_1d_to_plot = min(len(valid_1d_plots), ncols)
    for i in range(num_1d_to_plot):
        res_col, param_col = valid_1d_plots[i]
        print(f"    Plotting {clean_col_name(res_col)} vs {clean_col_name(param_col)} at index {plot_idx}")
        _add_1d_plot_to_dashboard(axes_flat[plot_idx], df.copy(), res_col, param_col, result_cols_info)
        plot_idx += 1
    for i in range(plot_idx, ncols):
        axes_flat[i].text(0.5, 0.5, "(Empty Slot)", ha='center', va='center', fontsize=10, color='grey')
        axes_flat[i].axis('off')
        plot_idx += 1

    # Add Combined Tables
    table_ax_idx = plot_idx
    print("  Adding combined tables...")
    _add_combined_tables_to_dashboard(axes_flat[table_ax_idx], df.copy(),
                                      valid_min_tables, all_result_formats, param_cols_config)
    plot_idx += 1

    # Add Minimum Sensitivity ASD Plot
    asd_ax_idx = plot_idx
    print("  Adding minimum sensitivity ASD plot...")
    _add_asd_plot_to_dashboard(axes_flat[asd_ax_idx], asd_filepath)
    plot_idx += 1

    # Add 2D Plot
    plot2d_ax_idx = plot_idx
    print("  Adding 2D plot...")
    if valid_2d_plot:
        res_col, (p1, p2) = valid_2d_plot
        print(
            f"    Plotting {clean_col_name(res_col)} vs {clean_col_name(p1)} & {clean_col_name(p2)} at index {plot2d_ax_idx}")
        _add_2d_plot_to_dashboard(axes_flat[plot2d_ax_idx], df.copy(), res_col, p1, p2, result_cols_info)
    else:
        axes_flat[plot2d_ax_idx].text(0.5, 0.5, "(No 2D Plot)", ha='center', va='center', fontsize=10, color='grey')
        axes_flat[plot2d_ax_idx].axis('off')
    plot_idx += 1

    # Finalize Dashboard Figure
    fig.suptitle('Experiment Parameter Scan Dashboard', fontsize=16)

    try:
        fig.subplots_adjust(
            left=0.08,
            right=0.95,
            bottom=0.1,
            top=0.92,
            wspace=0.3,
            hspace=0.4
        )
        print("Applied fig.subplots_adjust for layout.")
    except Exception as e_adjust:
        print(f"Warning: Could not apply fig.subplots_adjust: {e_adjust}")
        try:
            print("Falling back to plt.tight_layout...")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        except Exception as e_fallback:
            print(f"Fallback tight_layout also failed: {e_fallback}")

    dashboard_filepath = os.path.join(output_dir, f"dashboard.{plot_format}")
    try:
        plt.savefig(dashboard_filepath, format=plot_format, bbox_inches='tight', dpi=150)
        print(f"Saved Dashboard: {dashboard_filepath}")
    except Exception as e:
        print(f"Error saving dashboard figure {dashboard_filepath}: {e}")
    plt.close(fig)


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Loading data from: {FILE_PATH}")
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}");
        exit()

    # Set raw data directory
    RAW_DATA_DIR = os.path.dirname(os.path.abspath(FILE_PATH)) if os.path.dirname(FILE_PATH) else '.'
    print(f"Raw data directory set to: {RAW_DATA_DIR}")

    try:
        # Read CSV
        try:
            df_original = pd.read_csv(FILE_PATH, sep=None, engine='python', comment='#', skipinitialspace=True)
        except ValueError:
            print("Separator detection failed, trying common separators...")
            for sep in [',', '\t', r'\s+']:
                try:
                    df_original = pd.read_csv(FILE_PATH, sep=sep, comment='#', skipinitialspace=True); print(
                        f"  Successfully read with separator: '{sep}'"); break
                except Exception:
                    continue
            else:
                raise ValueError("Could not determine separator or read CSV.")

        print(f"Original columns found: {df_original.columns.tolist()}")
        df_original.columns = df_original.columns.str.strip()
        df_original.rename(columns=lambda x: x.strip(), inplace=True)

        # Create a working copy for processing/rounding
        df = df_original.copy()

        # Check for required columns and filter config if needed
        all_config_cols = set(PARAMETER_COLS) | set(RESULT_COLS.keys()) | set(MIN_TABLE_RESULTS.keys()) | {
            MIN_SENS_RESULT_COL}
        for r, p in DASHBOARD_1D_PLOTS_SPEC: all_config_cols.add(r); all_config_cols.add(p)
        if DASHBOARD_2D_PLOT_SPEC: r, (p1, p2) = DASHBOARD_2D_PLOT_SPEC; all_config_cols.add(r); all_config_cols.add(
            p1); all_config_cols.add(p2)

        # Check against the actual columns in the dataframe
        missing_cols = [col for col in all_config_cols if col not in df.columns]
        if missing_cols:
            print(
                f"\nWarning: The following configured columns are missing from the CSV: {sorted(list(set(missing_cols)))}")
            # Update configuration lists based on available columns
            PARAMETER_COLS = [p for p in PARAMETER_COLS if p in df.columns]
            RESULT_COLS = {k: v for k, v in RESULT_COLS.items() if k in df.columns}
            MIN_TABLE_RESULTS = {k: v for k, v in MIN_TABLE_RESULTS.items() if k in df.columns}
            ALL_RESULT_COLS_FOR_TABLES = {k: v for k, v in ALL_RESULT_COLS_FOR_TABLES.items() if k in df.columns}
            DASHBOARD_1D_PLOTS_SPEC = [(r, p) for r, p in DASHBOARD_1D_PLOTS_SPEC if
                                       r in df.columns and p in df.columns]
            if DASHBOARD_2D_PLOT_SPEC:
                res, (p1, p2) = DASHBOARD_2D_PLOT_SPEC
                if not (res in df.columns and p1 in df.columns and p2 in df.columns): DASHBOARD_2D_PLOT_SPEC = None
            if MIN_SENS_RESULT_COL not in df.columns:
                print(
                    f"  ** Critical Warning: Minimum sensitivity column '{MIN_SENS_RESULT_COL}' for ASD plot is missing! **")
                MIN_SENS_RESULT_COL = None

            if not PARAMETER_COLS: print(
                "Error: No configured parameter columns found in file after filtering. Exiting."); exit()
            print("  Configurations updated to use only available columns.")

        print("\n--- Applying Initial Conversions and Rounding (to working copy) ---")
        # Identify all columns potentially needing numeric conversion or rounding
        cols_to_process_numeric = list(set(PARAMETER_COLS) | set(RESULT_COLS.keys()) | set(MIN_TABLE_RESULTS.keys()))
        if MIN_SENS_RESULT_COL: cols_to_process_numeric.append(MIN_SENS_RESULT_COL)
        cols_to_process_numeric = list(set(cols_to_process_numeric))

        for col in cols_to_process_numeric:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    nan_before = df[col].isnull().sum()
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    nan_after = df[col].isnull().sum()
                    print(f"  Converted '{col}' to numeric. Introduced {nan_after - nan_before} NaNs.")

                # Apply rounding based on PARAM_ROUNDING config
                if col in PARAM_ROUNDING and pd.api.types.is_numeric_dtype(df[col]):
                    precision = PARAM_ROUNDING[col]
                    nan_before = df[col].isnull().sum()
                    if precision < 0:  # Round to nearest multiple
                        base = 10 ** abs(precision)
                        df[col] = round_to_nearest(df[col].copy(), base)
                        print(f"  Rounded Parameter '{col}' to nearest {int(base)}.")
                    else:  # Round to decimal places
                        df[col] = df[col].round(precision)
                        print(f"  Rounded Parameter '{col}' to {precision} decimal places.")
                    nan_after = df[col].isnull().sum()
                    if nan_after > nan_before:
                        print(f"    Warning: Rounding introduced {nan_after - nan_before} NaNs in '{col}'.")

        print("Initial processing complete.")
    except FileNotFoundError:
        print(f"Error: File not found at {FILE_PATH}");
        exit()
    except ValueError as e:
        print(f"Error reading or parsing CSV file: {e}");
        exit()
    except Exception as e:
        print(f"Error during initial loading or processing: {type(e).__name__} - {e}")
        import traceback

        traceback.print_exc()
        exit()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR} (Format: {PLOT_FORMAT})")

    # Analyze Parameter Variation
    varying_params = []
    print("\n--- Analyzing Parameter Variation (Post-Rounding/Conversion) ---")
    for col in PARAMETER_COLS:
        if col in df.columns:
            unique_values = df[col].dropna().unique()
            n_unique = len(unique_values)
            # Format examples using original values
            try:
                original_unique = df_original[col].dropna().unique()
                display_vals = sorted(list(original_unique))[:5]
                formatted_vals = [format_parameter_value_for_table(col, v) for v in display_vals]
                display_str = ", ".join(formatted_vals)
            except Exception:
                display_str = ", ".join(map(str, sorted(list(unique_values))[:5]))
            if n_unique > 5: display_str += "..."
            print(
                f"Parameter '{clean_col_name(col)}': {n_unique} unique non-NaN value(s) after rounding. Examples (Original): [{display_str}]")
            if n_unique > 1: varying_params.append(col)

    if not varying_params:
        print("\nNo varying parameters found after rounding. Limited plots will be generated.")
    else:
        print(f"Varying parameters identified for plotting: {[clean_col_name(p) for p in varying_params]}")

    # Use the processed DataFrame (df) for plotting
    df_plot = df

    # Generate Individual 1D Plots
    print("\n--- Generating 1D Plots ---")
    if varying_params and RESULT_COLS:
        for res_col, info in RESULT_COLS.items():
            if res_col not in df_plot.columns: continue
            for param_col in varying_params:
                plot_1d_dependence(df_plot.copy(), param_col, {res_col: info}, OUTPUT_DIR, plot_format=PLOT_FORMAT)
    elif not varying_params:
        print("Skipping 1D plots: No varying parameters.")
    else:
        print("Skipping 1D plots: No result columns configured or found.")

    # Generate Individual 2D Plots
    print("\n--- Generating 2D Plots ---")
    if len(varying_params) >= 2 and RESULT_COLS:
        param_pairs = list(itertools.combinations(varying_params, 2))
        for res_col, info in RESULT_COLS.items():
            if res_col not in df_plot.columns: continue
            for param1, param2 in param_pairs:
                plot_2d_dependence(df_plot.copy(), param1, param2, {res_col: info}, OUTPUT_DIR, plot_format=PLOT_FORMAT)
    elif len(varying_params) < 2:
        print("Skipping 2D plots: Need at least 2 varying parameters.")
    else:
        print("Skipping 2D plots: No result columns configured or found.")

    # Create Dashboard
    create_dashboard(df_plot.copy(), df_original.copy(),
                     varying_params, RESULT_COLS,
                     DASHBOARD_1D_PLOTS_SPEC,
                     DASHBOARD_2D_PLOT_SPEC,
                     MIN_TABLE_RESULTS,
                     ALL_RESULT_COLS_FOR_TABLES,
                     PARAMETER_COLS,
                     OUTPUT_DIR, plot_format=PLOT_FORMAT)

    print("\nAnalysis complete.")