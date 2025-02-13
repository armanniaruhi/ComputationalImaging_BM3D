import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import yaml

# Initialize the Dash app
app = dash.Dash(__name__)

# File path and DataFrame loading
base_dir = os.getcwd()  # Current working directory

# Define all parameters in a dictionary and read from json
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)
    
params_block_ht= config["params_block_ht"]
params_block_wiener= config["params_block_wiener"]
params_patch_ht= config["params_patch_ht"]
params_patch_wiener= config["params_patch_wiener"]
params_step_ht= config["params_step_ht"]
params_step_wiener= config["params_step_wiener"]
params_threshold_ht= config["params_threshold_ht"]
params_threshold_wiener= config["params_threshold_wiener"]

# Function to process the DataFrame
def process_dataframe(fix_y, fix_title_value, fix_transform, params, dataset_dropdown):
    file_path = f"{base_dir}{params['file_path']}"
    try:
        file_path = file_path.replace("Leech", dataset_dropdown)
    except Exception as e:
        file_path = file_path.replace("resTarget", dataset_dropdown)
    df_block = pd.read_excel(file_path, header=0)
    
    # Extract parameters
    fix_column_header = params["fix_column_header"]
    renamed_column_subheader = params["renamed_column_subheader"]
    selected_cols = params["selected_cols"]

    # Filter the DataFrame based on conditions
    df_temp = df_block[df_block["Coordinate y"] == fix_y]
    df_temp = df_temp[df_temp[fix_column_header] == fix_title_value]
    df_temp = df_temp[df_temp["Transform 2D"] == fix_transform]

    # Reset index and select columns based on the selected_cols dictionary
    df_temp.reset_index(inplace=True)
    df_temp1 = df_temp[list(selected_cols.keys())]

    # Rename columns using the dictionary values
    df_temp1 = df_temp1.rename(columns=selected_cols)

    # Group by 'Sigma' and renamed_column_subheader, and compute the mean
    grouped = df_temp1.groupby([renamed_column_subheader, "Sigma"]).mean().reset_index()

    # Pivot the table to create multi-level columns
    pivoted = grouped.pivot(index="Sigma", columns=renamed_column_subheader)

    # Swap the levels to bring renamed_column_subheader to the first level and Sigma to the second
    pivoted.columns = pivoted.columns.swaplevel(0, 1)

    # Sort the columns to group by renamed_column_subheader and round the values
    pivoted = pivoted.sort_index(axis=1, level=0).round(2)

    # Reset index so that Sigma becomes a column rather than the index
    pivoted.reset_index(inplace=True)

    # Rename columns for better readability (set multi-index names)
    pivoted.columns.names = [renamed_column_subheader, "Sigma"]

    # ----- Compute summary statistics and append as new rows -----
    summary_labels = ['Mean', 'Median']  # Removed 'Mode' and 'Std'
    # Initialize a dictionary to hold summary rows.
    summary_rows = {col: [] for col in pivoted.columns}

    # Iterate over each column in pivoted to compute the summary statistics.
    for col in pivoted.columns:
        if col == ("Sigma",""):
            # For the 'Sigma' column, put the summary label for each row.
            for label in summary_labels:
                summary_rows[col].append(label)
        else:
            # Get the numeric data for the column, skipping any NaNs.
            data = pivoted[col].dropna()
            # Compute the statistics; if data is empty, set to np.nan.
            mean_val = round(data.mean(), 2) if not data.empty else np.nan
            median_val = round(data.median(), 2) if not data.empty else np.nan
            summary_rows[col].extend([mean_val, median_val])  # Only mean and median

    # Create a DataFrame from the summary rows.
    summary_df = pd.DataFrame(summary_rows, index=summary_labels).reset_index(drop=True)

    # Append the summary rows to the pivoted DataFrame.
    pivoted = pd.concat([pivoted, summary_df], ignore_index=True)

    return pivoted, renamed_column_subheader


# Layout of the app (unchanged)
app.layout = html.Div([
    html.H1("BM3D Filtering result analysis"),
    
    # Dropdown for selecting alternative parameter
    html.Div([
        html.Label("Select Dataset:"),
        dcc.Dropdown(
            id='dataset_dropdown',
            options=[
                {'label': 'Leech', 'value': 'Leech'},
                {'label': 'resTarget', 'value': 'resTarget'},
            ],
            value='Leech',  # Default value
            style={'width': '50%'}
        ),
    ]),
    
    # Dropdown for selecting alternative parameter
    html.Div([
        html.Label("Select Parameters Set:"),
        dcc.Dropdown(
            id='params_dropdown',
            options=[
                {'label': 'Fix Patch Hard', 'value': 'params_patch_ht'},
                {'label': 'Fix Patch Wiener', 'value': 'params_patch_wiener'},
                {'label': 'Fix Block Hard', 'value': 'params_block_ht'},
                {'label': 'Fix Block Wiener', 'value': 'params_block_wiener'},
                {'label': 'Fix Step Hard', 'value': 'params_step_ht'},
                {'label': 'Fix Step Wiener', 'value': 'params_step_wiener'},
                {'label': 'Fix Threshold Hard', 'value': 'params_threshold_ht'},
                {'label': 'Fix Threshold Wiener', 'value': 'params_threshold_wiener'}
            ],
            style={'width': '50%'},
            value='Fix Patch Hard'  # Default value
        ),
    ]),
    
    
    # Dropdown for selecting fix_title_value
    html.Div([
        html.Label("Value of fix parmeter:"),
        dcc.Dropdown(
            id='fix_title_value_dropdown',
            value=18,
            style={'width': '50%'}
        ),
    ]),

    # Dropdown for selecting fix_y value
    html.Div([
        html.Label("Select Image:"),
        dcc.Dropdown(
            id='fix_y_dropdown',
            options=[{'label': f'Y = {y}', 'value': y} for y in range(15, 23)],
            style={'width': '50%'},
            value=16
        ),
    ]),
    # Dropdown for selecting fix_transform
    html.Div([
        html.Label("Select type of 2D-Transform:"),
        dcc.Dropdown(
            id='fix_transform_dropdown',
            options=[{'label': 'bior1.5', 'value': 'bior1.5'},
                     {'label': 'dct', 'value': 'dct'}],
            value='bior1.5',  # Default value for fix_transform
            style={'width': '50%'}
        ),
    ]),
    
    # Div to display the table
    dcc.Graph(id='table')
])

# Callback to update the fix_title_value dropdown options dynamically based on fix_y selection
@app.callback(
    Output('fix_title_value_dropdown', 'options'),
    [Input('fix_y_dropdown', 'value'),
     Input('params_dropdown', 'value')]
)
def update_value_dropdown(fix_y, params_key):
    params_dict = globals()[params_key]  # Dynamically get the parameters based on dropdown value
    
    # Based on selected fix_y, update the options for the fix_title_value dropdown
    options = [{'label': str(value), 'value': value} for value in params_dict["fix_value"]]
    return options

# Callback to update table based on dropdown selection
@app.callback(
    Output('table', 'figure'),
    [Input('fix_y_dropdown', 'value'),
     Input('fix_title_value_dropdown', 'value'),
     Input('fix_transform_dropdown', 'value'),
     Input('params_dropdown', 'value'),
     Input('dataset_dropdown', 'value')]
)
def update_table(fix_y, fix_title_value, fix_transform, params_key, dataset_dropdown):
    params_dict = globals()[params_key]  # Dynamically get the parameters based on dropdown value
    
    # Generate the table based on the selected values
    pivoted, renamed_column_subheader = process_dataframe(
        fix_y, fix_title_value, fix_transform, params_dict, dataset_dropdown
    )
    
    # Pre-initialize cell colors for every cell as white.
    cell_colors = [['white'] * len(pivoted) for _ in pivoted.columns]
    
    # Iterate through each column (using the multi-index columns)
    for col_num, col_ in enumerate(pivoted.columns):
        col = col_[1]
        column_data = pivoted.iloc[:, col_num]
        total_rows = len(column_data)
        # Adjusted to account for 2 summary rows (Mean and Median)
        valid_index = total_rows - 2 if total_rows > 2 else total_rows
        valid_data = column_data.iloc[:valid_index]  # rows to be considered
        
        bold_flags = [False] * total_rows

        if valid_data.empty:
            pass
        else:
            if "BRISQUE" in col:
                target_val = valid_data.min()
            elif "Clip-IQA" in col:
                target_val = valid_data.max()
            elif "TV" in col:
                target_val = valid_data.min()
            elif "SNR" in col:
                target_val = valid_data.max()
            else:
                target_val = None

            if target_val is not None:
                for i in range(valid_index):
                    if column_data.iloc[i] == target_val:
                        bold_flags[i] = True

        # Update cell colors
        for i in range(total_rows):
            if i >= valid_index:
                cell_colors[col_num][i] = 'lightgrey'
            else:
                if bold_flags[i]:
                    if ("SNR" in col) or ("Clip-IQA" in col):
                        cell_colors[col_num][i] = 'lightgreen'
                    else:
                        cell_colors[col_num][i] = 'orange'
    
    # Create Plotly table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(pivoted.columns),
            fill_color='lightgray',
            align='center',
            font=dict(size=12)
        ),
        cells=dict(
            values=[pivoted[col] for col in pivoted.columns],
            fill_color=cell_colors,
            align='center',
            font=dict(size=14)
        ),
        columnwidth=[0.5] * len(pivoted.columns)
    )])
    fig.update_layout(
    title=None,
    annotations=[
        dict(
            text=f"<b>(Y :</b> {fix_y}) , <b>(Value :</b> {fix_title_value}), <b>(2D-Transform :</b> {fix_transform})<br>"
                 f"In the first row you could see all values of <b><span style='color:red;'>{renamed_column_subheader}</span></b>",
            x=0.5,
            y=-0.2,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16),
            align="center"
        )
    ]
)

    
    return fig


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)