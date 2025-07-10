# main.py
import sys
from PyQt6.QtWidgets import QApplication
from startup_loader import start_main_application

def main(app_instance, file=None, channel_index=None):
    return start_main_application(app_instance, file, channel_index)

if __name__ == "__main__":
    main()