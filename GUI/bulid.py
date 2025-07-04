import subprocess

subprocess.run([
    r'.venv\Scripts\python.exe',
    '-m', 'PyInstaller',
    './GUI/main.py',
    '--name=SCIDO',
    '--distpath=./BUILD',
    '--onedir',
    '--console',
    '--icon=GUI/SCIDO.ico',
    '--add-data=GUI/icons;icons',
    '--add-data=GUI/fonts;fonts',
    '--add-data=GUI/models;models',
])