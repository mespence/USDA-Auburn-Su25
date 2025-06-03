# when some criteria is met:
#   read new data from csv (can i make it only read the last few lines?)
#   put the new data somewhere (buffer, another file, straight to screen, ???)

#       assuming this is put into the gui, also need to check whether we're in scroll mode or explore mode
# actually im making this a scrap here:
scroll_mode = True
if scroll_mode:
    #do the thing below
    # mouseDragEvent seems like it might be useful? cause main two things are making sure it auto updates and making it switch modes upon interaction
    pass
else:
    #do the preexisting code (I think the stuff in DataWindow2?)
    pass
    


import time

csv_path = "testdata1.csv"

def follow_csv(csv_path):
    with open(csv_path, "r") as f:
        # Move to the end of file
        # f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                # this assumes we want to check on a schedule rather than every time the csv is updated
                time.sleep(0.5)  # Wait for new data
                continue
            print(line.strip())  # Replace with your processing logic

follow_csv(csv_path)