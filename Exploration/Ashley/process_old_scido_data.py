import pandas as pd
import numpy as np
import os
import sys
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

def process_old_scido_data():
    """
    Uses QFileDialog to let the user select an input CSV file.
    Then, it prompts the user for a new output filename (which will be saved
    in the same directory as the input file), processes the data,
    and saves the new CSV.
    """
    app = None
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()

    input_filename, _ = QFileDialog.getOpenFileName(
        None,
        "Select Input CSV File",
        "", # use default working directory or the last opened
        "CSV Files (*.csv);;All Files (*)"
    )

    if not input_filename:
        QMessageBox.warning(None, "No File Selected", "No input file was selected. Exiting the Program.")
        if app: # Quit if we created the QApplication instance
            app.quit()
        return

    # Determine the directory of the input file
    input_directory = os.path.dirname(input_filename)
    original_basename = os.path.splitext(os.path.basename(input_filename))[0]

    # 2. Prompt for the output file name (just the base name)

    output_filename, _ = QFileDialog.getSaveFileName(
        None,
        "Save Processed Data As",
        os.path.join(input_directory, f"{original_basename}_processed.csv"), # Suggests a name in the same dir
        "CSV Files (*.csv);;All Files (*)"
    )

    if not output_filename:
        QMessageBox.warning(None, "File Not Saved", "No output filename was provided. Exiting the Program.")
        if app:
            app.quit()
        return

    # Ensure the output filename has a .csv extension
    if not output_filename.lower().endswith('.csv'):
        output_filename += '.csv'

    try:
        df = pd.read_csv(input_filename)
        new_df = pd.DataFrame(columns=["time", "voltage", "labels", "comments"])
        print(f"Successfully loaded '{input_filename}'. Columns found: {df.columns.tolist()}")

        # Ensure 'time', 'labels' and 'comments' columns exist, add if missing
        if 'time' not in df.columns:
            QMessageBox.critical(None, "Missing Column", "'time' column not found in the input file. This column is required. Please upload another file.")
            if app:
                app.quit()
            return
        else:
            new_df['time'] = df['time']
        
        if 'labels' in df.columns:
            new_df['labels'] = df['labels']
        else:
            new_df['labels'] = np.nan

        if 'comments' in df.columns:
            new_df['comments'] = df['comments']
        else:
            new_df['comments'] = np.nan

        # assign voltage column
        available_voltage_cols = []
        if 'pre_rect' in df.columns:
            available_voltage_cols.append('pre_rect')
        if 'post_rect' in df.columns:
            available_voltage_cols.append('post_rect')

        if not available_voltage_cols:
            QMessageBox.critical(None, "Missing Voltage Data", "Neither 'pre_rect' nor 'post_rect' columns found in the CSV. Cannot create the 'voltage' column. Please upload another file.")
            if app:
                app.quit()
            return

        choice = None
        if len(available_voltage_cols) == 1:
            choice = available_voltage_cols[0]
            QMessageBox.information(None, "Voltage Column Selection", f"Only '{choice}' found. Using it as 'voltage'.")
        else:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Voltage Column Selection")
            msg_box.setText("Which column would you like to use as 'voltage'?")

            post_rect_btn = msg_box.addButton("post_rect", QMessageBox.ButtonRole.ActionRole)
            pre_rect_btn = msg_box.addButton("pre_rect", QMessageBox.ButtonRole.ActionRole)
            msg_box.exec()

            if msg_box.clickedButton() == pre_rect_btn:
                choice = 'pre_rect'
            elif msg_box.clickedButton() == post_rect_btn:
                choice = 'post_rect'
            else:
                QMessageBox.warning(None, "Selection Cancelled", "No column selected for 'voltage'. Exiting the Program.")
                if app:
                    app.quit()
                return

        # Rename the selected column to 'voltage'
        new_df['voltage'] = df[choice]
                
        print(f"Column '{choice}' has been renamed to 'voltage'.")

        new_df.to_csv(output_filename, index=True)
        QMessageBox.information(None, "Success", f"Processed data saved successfully to '{output_filename}'.")

    except pd.errors.EmptyDataError:
        QMessageBox.critical(None, "Error", f"The input file '{input_filename}' is empty.")
    except pd.errors.ParserError:
        QMessageBox.critical(None, "Error", f"Could not parse '{input_filename}'. Please ensure it's a valid CSV.")
    except Exception as e:
        QMessageBox.critical(None, "An Error Occurred", f"An unexpected error occurred during processing: {e}")
    finally:
        if app:
            app.quit() # Ensure the QApplication is properly shut down

if __name__ == "__main__":
    process_old_scido_data()