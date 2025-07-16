import subprocess
import os

icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'icons'))
font_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fonts'))
logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'SCIDO.png'))
ico_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'SCIDO.ico'))

subprocess.run([
    r"C:\Users\jomo\AppData\Local\Programs\Python\Python311\python.exe",
    '-m', 'PyInstaller',
    './AppLauncherDialog.py',
    '--name=SCIDO',
    '--workpath=./build/.scido_temp',
    '--distpath=./build',
    '--onedir',
    '--console',
    '--icon=./SCIDO.ico',
    '--exclude-module=PyQt5',
    '--exclude-module=PySide6',
    f'--add-data={icon_path}:icons',
    f'--add-data={font_path}:fonts',
    f'--add-data={logo_path}:.',
    f'--add-data={ico_path}:.',
    '--noconfirm',
])