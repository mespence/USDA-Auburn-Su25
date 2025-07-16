import windaq

filepath = r"C:\Users\jomo\Downloads\AeA on hand FPSO 11July2025 no01.WDH"

daq = windaq.windaq(filepath)

print(daq.time())