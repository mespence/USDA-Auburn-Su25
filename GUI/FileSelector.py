import os
import re
#import sys

from PyQt6.QtWidgets import QFileDialog

from DataWindow import DataWindow
from EPGData import EPGData

class FileSelector:
    def load_new_data(epgdata: EPGData, datawindow) -> None:
        datawindow.transition_mode = 'labels'
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileUrl()
        file_path = file_path.toLocalFile()
        if file_path:
            if epgdata.load_data(file_path):
                datawindow.plot_recording(file_path)
                if isinstance(datawindow, DataWindow):
                     datawindow.plot_transitions(file_path)
                #datawindow.mode = 'labels'
                #datawindow.plot_comments(file_path)

    def export_labeled_data(epgdata: EPGData, file: str):
        file_dialog = QFileDialog()
        file_dialog.AcceptMode = 1 # save mode
        file_url, selected_filter = file_dialog.getSaveFileUrl(filter="CSV files (*.csv);;text files (*.txt)")
        file_path = file_url.toLocalFile()
        if selected_filter:
            extension = re.match(r'^.*\(\*\.(.*)\)$', selected_filter).group(1)
        if file_path:
            dirname  = os.path.dirname(file_path)
            basename = os.path.basename(file_path)
            # NOTE: not sure if it's a misuse of the filter to use it to force the extension,
            # rather than just using it to limit what files are visible in the dialog
            if '.' in basename:
                if basename.rsplit(".", 1)[1] != extension:
                    basename = basename.rsplit(".", 1)[0] + '.' + extension
            else:
                basename += '.' + extension
            full_path = os.path.join(dirname, basename)
            if extension.lower() == 'csv':
                epgdata.export_csv(file, full_path)
            elif extension.lower() == 'txt':
                epgdata.export_txt(file, full_path)

